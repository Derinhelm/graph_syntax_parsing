from itertools import chain
from operator import itemgetter
from torch_geometric.loader import DataLoader
import tqdm

from errors import ErrorInfo
from gnn import GNNNet
from scores import TrainScores, TestScores

class Oracle:
    def __init__(self, options, irels, device):
        self.net = GNNNet(options, len(irels), device)
        self.irels = irels
        self.error_info = ErrorInfo()

    def create_test_transition(self, config_to_predict_list):
        best_transition_list = []
        config_list = [config for config, _, _, _, _ in config_to_predict_list]
        all_scrs_list = []
        #print("config_list len:", len(config_list))
        graph_info_list = [config.graph.get_graph() for config in config_list]
        graph_loader = DataLoader(graph_info_list, batch_size=self.net.elems_in_batch, shuffle=False)
        pbar = tqdm.tqdm(
            graph_loader,
            desc="Batch processing",
            unit="batch",
            mininterval=1.0,
            leave=False,
        )
        for batch in graph_loader:
            cur_all_scrs, _ = self.net.evaluate(batch)
            all_scrs_list += cur_all_scrs
        for i, all_scrs in enumerate(all_scrs_list):
            config, _, max_swap, _, iSwap = config_to_predict_list[i]
            scrs, uscrs = self.net.get_scrs_uscrs(all_scrs)
            scores_info = TestScores(scrs, uscrs)
            scores = scores_info.test_evaluate(config, self.irels)
            best = max(chain(*(scores if iSwap < max_swap else scores[:3] )), key = itemgetter(2) )
            best_transition_list.append(best)
        return best_transition_list

    def create_train_transition_batch(self, batch, batch_config_list, dynamic_oracle):
        best_transition_list = []
        non_detach_cur_all_scrs, cur_all_scrs = self.net.evaluate(batch)
        for i, all_scrs in enumerate(cur_all_scrs):
            config = batch_config_list[i]
            scrs, uscrs = self.net.get_scrs_uscrs(all_scrs)
            non_detach_all_scrs = non_detach_cur_all_scrs[i]
            non_detach_scrs, non_detach_uscrs = self.net.get_scrs_uscrs(non_detach_all_scrs)
            scores_info = TrainScores(scrs, uscrs, non_detach_scrs, non_detach_uscrs)
            best, shift_case = \
                scores_info.create_best_transaction(config, dynamic_oracle,
                                                    self.error_info, self.irels)
            best_transition_list.append((best, shift_case))
        return best_transition_list


    def error_processing(self, is_final):
        #if self.error_info.processing_check(is_final):
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

    def change_sentence_number(self, iSentence):
        self.error_info.change_sentence_number(iSentence)
