from itertools import chain
from operator import itemgetter
from torch_geometric.loader import DataLoader
import tqdm

from errors import ErrorInfo
from scores import TrainScores, TestScores

class Oracle:
    def __init__(self, options, irels, device, mode):
        self.mode = mode
        if mode == "graph":
            from gnn import GNNNet
            self.net = GNNNet(options, len(irels), device)
        else:
            from mlp import MLPNet
            self.net = MLPNet(options, len(irels), device)
        print(self.net)
        self.elems_in_batch = options["elems_in_batch"]
        self.irels = irels
        self.error_info = ErrorInfo()

    def create_test_transition_batch(self, batch, batch_config_param_list):
        cur_all_scrs, _ = self.net.evaluate(batch)
        batch_best_transition_list = []
        for i, all_scrs in enumerate(cur_all_scrs):
            config, _, max_swap, _, iSwap = batch_config_param_list[i]
            scrs, uscrs = self.net.get_scrs_uscrs(all_scrs)
            scores_info = TestScores(scrs, uscrs)
            scores = scores_info.test_evaluate(config, self.irels)
            best = max(chain(*(scores if iSwap < max_swap else scores[:3] )), key = itemgetter(2) )
            batch_best_transition_list.append(best)
        return batch_best_transition_list

    def create_test_transition(self, config_param_list, device):
        best_transition_list = []
        config_list = [config for config, _, _, _, _ in config_param_list]
        config_embed_list = [config.get_config_embed(device, self.mode) for config in config_list]
        config_embed_loader = DataLoader(
            config_embed_list, batch_size=self.elems_in_batch, shuffle=False)
        pbar = tqdm.tqdm(
            config_embed_loader,
            desc="Batch processing",
            unit="batch",
            mininterval=1.0,
            leave=False,
        )
        config_i = 0
        for batch in config_embed_loader:
            batch_config_param_list = \
                    config_param_list[config_i:config_i + len(batch)]
            config_i += len(batch)
            best_transition_batch_list = \
                self.create_test_transition_batch(batch, batch_config_param_list)
            best_transition_list += best_transition_batch_list

        return best_transition_list

    def create_score_structure(self, net_res_i):
        non_detach_all_scrs, all_scrs = net_res_i
        scrs, uscrs = self.net.get_scrs_uscrs(all_scrs)
        non_detach_scrs, non_detach_uscrs = self.net.get_scrs_uscrs(non_detach_all_scrs)
        scores_info = TrainScores(scrs, uscrs, non_detach_scrs, non_detach_uscrs)
        return scores_info

    def create_train_transition_batch(self, batch, batch_config_list, dynamic_oracle):
        best_transition_list = []
        net_res = self.net.evaluate(batch)
        for i, net_res_i in enumerate(zip(*net_res)):
            config = batch_config_list[i]
            scores_info = self.create_score_structure(net_res_i)
            best, shift_case = scores_info.create_best_transition(
                config, dynamic_oracle, self.error_info, self.irels)
            best_transition_list.append((best, shift_case))
        return best_transition_list


    def error_processing(self, is_final):
        errs = self.error_info.get_errs()
        self.net.error_processing(errs)
        self.error_info.set_errs()

    def Load(self, epoch):
        self.net.Load(epoch)

    def Save(self, epoch):
        self.net.Save(epoch)

    def train_logging(self):
        self.error_info.train_logging()

    def get_mloss(self):
        return self.error_info.get_mloss()

    def change_sentence_number(self, sentence_ind):
        self.error_info.change_sentence_number(sentence_ind)
