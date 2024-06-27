import copy
from logging import getLogger
import random
import time
import torch
from torch_geometric.loader import DataLoader
import tqdm

from configuration import Configuration
from constants import SWAP
from oracle import Oracle
from utils import ConllEntry

class Parser:
    def __init__(self, options, irels, embeds):
        self.dynamic_oracle = options["dynamic_oracle"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        print("device:", self.device)
        self.oracle = Oracle(options, irels, self.device)
        self.embeds = embeds
        self.info_logger = getLogger('info_logger')
        self.time_logger = getLogger('time_logger')
        #self.transition_logger = getLogger('transition_logger')

    def Load(self, epoch):
        self.oracle.Load(epoch)

    def Save(self, epoch):
        self.oracle.Save(epoch)

    def test_transition_processing(self, config, best, max_swap, sentence_ind, \
                                    reached_max_swap, reached_swap_for_i_sentence, iSwap):
        if iSwap == max_swap and not reached_swap_for_i_sentence:
            reached_max_swap += 1
            reached_swap_for_i_sentence = True
            self.info_logger.debug(f"reached max swap in {reached_max_swap:d} \
                            out of {sentence_ind:d} sentences")
        config.apply_transition(best)
        if best[1] == SWAP:
            iSwap += 1
        return reached_max_swap, reached_swap_for_i_sentence, iSwap

    def create_conll_res(self, osentence, config):
        #keep in memory the information we need, not all the vectors
        oconll_sentence = [entry for entry in osentence if isinstance(entry, ConllEntry)]
        oconll_sentence = oconll_sentence[1:] + [oconll_sentence[0]]
        conll_sentence = config.get_sentence()
        for tok_o, tok in zip(oconll_sentence, conll_sentence):
            tok_o.pred_relation = tok.pred_relation
            tok_o.pred_parent_id = tok.pred_parent_id
        return osentence

    def Predict(self, data):
        self.oracle.net.gnn.eval()
        reached_max_swap = 0
        config_list = []
        isentence_config_dict = {} # sentence_ind -> Configuration
        for sentence_ind, osentence in enumerate(data,1):
            config = Configuration(osentence, self.oracle.irels, self.embeds, self.device)
            # конфигурации в порядке предложений в данных, в процессе работы конфигурации изменяются
            # в итоге все конфигурации станут итоговыми
            isentence_config_dict[sentence_ind] = config
            max_swap = 2*len(osentence)
            reached_swap_for_i_sentence = False
            iSwap = 0
            config_list.append((config, sentence_ind, max_swap, \
                                          reached_swap_for_i_sentence, iSwap))

        while len(config_list) != 0:
            new_config_list = []
            best_config_list = \
                self.oracle.create_test_transition(config_list)

            for i, config_to_predict in enumerate(config_list):
                config, sentence_ind, max_swap, reached_swap_for_i_sentence, iSwap = config_to_predict
                best = best_config_list[i]
                reached_max_swap, reached_swap_for_i_sentence, iSwap = \
                        self.test_transition_processing(config, best, max_swap, sentence_ind,
                                            reached_max_swap, reached_swap_for_i_sentence, iSwap)
                if not config.is_end():
                    new_config_list.append((config, sentence_ind, max_swap, 
                                                      reached_swap_for_i_sentence, iSwap))
            config_list = new_config_list

        for sentence_ind, osentence in enumerate(data,1):
            config = isentence_config_dict[sentence_ind]
            res_osentence = self.create_conll_res(osentence, config)
            yield res_osentence

    def train_transition_processing(self, config, best, shift_case):
        #updates for the dynamic oracle
        if self.dynamic_oracle:
            config.dynamic_oracle_updates(best, shift_case)

        config.apply_transition(best)
        return

    def create_next_configs_batch(self, batch, batch_config_list):
        not_finished_configs = []
        best_transition_batch_list = \
            self.oracle.create_train_transition_batch(batch,
                                                        batch_config_list,
                                                        self.dynamic_oracle)
        for i, config in enumerate(batch_config_list):
            best_transition, shift_case = best_transition_batch_list[i]
            self.train_transition_processing(config, best_transition, shift_case)
            if not config.is_end():
                not_finished_configs.append(config)
        self.oracle.error_processing(False)
        return not_finished_configs

    def Train(self, trainData):
        self.oracle.net.gnn.train()
        random.shuffle(trainData)

        # in certain cases the data will already have been shuffled
        # after being read from file or while creating dev data
        self.info_logger.info(f"Length of training data: {len(trainData)}")

        beg = time.time()
        config_list = [Configuration(sentence, self.oracle.irels,
                                     self.embeds, self.device)
                            for sentence in trainData]

        iter_num = 0
        while len(config_list) != 0:
            print(iter_num, len(config_list))
            config_embed_list = [config.get_config_embed()
                               for config in config_list]
            config_embed_loader = DataLoader(
                config_embed_list, batch_size=self.oracle.elems_in_batch,
                shuffle=False)
            config_i, new_config_list = 0, []
            for batch in config_embed_loader:
                batch_config_list = \
                    config_list[config_i:config_i + len(batch)]
                config_i += len(batch)
                new_config_list += \
                    self.create_next_configs_batch(batch, batch_config_list)

            config_list = new_config_list
            iter_num += 1
        self.oracle.error_processing(True)
        mloss = self.oracle.get_mloss()

        self.info_logger.info(f"Loss: {mloss / len(trainData)}")
        self.info_logger.info(f"Total Training Time: {time.time() - beg:.2g}s")
