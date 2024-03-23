import copy
from logging import getLogger
import random
import time
import torch
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

    def test_transition_processing(self, config, best, max_swap, iSentence, \
                                    reached_max_swap, reached_swap_for_i_sentence, iSwap):
        if iSwap == max_swap and not reached_swap_for_i_sentence:
            reached_max_swap += 1
            reached_swap_for_i_sentence = True
            self.info_logger.debug(f"reached max swap in {reached_max_swap:d} \
                            out of {iSentence:d} sentences")
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
        reached_max_swap = 0
        config_to_predict_list = []
        isentence_config_dict = {} # iSentence -> Configuration
        for iSentence, osentence in enumerate(data,1):
            config = Configuration(osentence, self.oracle.irels, self.embeds, self.device)
            # конфигурации в порядке предложений в данных, в процессе работы конфигурации изменяются
            # в итоге все конфигурации станут итоговыми
            isentence_config_dict[iSentence] = config 
            max_swap = 2*len(osentence)
            reached_swap_for_i_sentence = False
            iSwap = 0
            config_to_predict_list.append((config, iSentence, max_swap, \
                                          reached_swap_for_i_sentence, iSwap))

        while len(config_to_predict_list) != 0:
            new_config_to_predict_list = []
            best_config_list = self.oracle.create_test_transition(config_to_predict_list)

            for i in range(len(config_to_predict_list)):
                config, iSentence, max_swap, reached_swap_for_i_sentence, iSwap = config_to_predict_list[i]
                best = best_config_list[i]
                reached_max_swap, reached_swap_for_i_sentence, iSwap = \
                        self.test_transition_processing(config, best, max_swap, iSentence,
                                            reached_max_swap, reached_swap_for_i_sentence, iSwap)
                if not config.is_end():
                    new_config_to_predict_list.append((config, iSentence, max_swap, 
                                                      reached_swap_for_i_sentence, iSwap))
            config_to_predict_list = new_config_to_predict_list

        for iSentence, osentence in enumerate(data,1):
            config = isentence_config_dict[iSentence]
            res_osentence = self.create_conll_res(osentence, config)
            yield res_osentence

    def train_transition_processing(self, config, best, shift_case):
        #updates for the dynamic oracle
        if self.dynamic_oracle:
            config.dynamic_oracle_updates(best, shift_case)

        config.apply_transition(best)
        return

    def Train(self, trainData):
        random.shuffle(trainData)

        # in certain cases the data will already have been shuffled after being read from file or while creating dev data
        self.info_logger.info(f"Length of training data: {len(trainData)}")

        beg = time.time()
        config_to_predict_list = []
        for sentence in trainData:
            config = Configuration(sentence, self.oracle.irels, self.embeds, self.device)
            config_to_predict_list.append(config)
        iter_num = 0
        while len(config_to_predict_list) != 0:
            print(iter_num, len(config_to_predict_list))
            best_transition_list = self.oracle.create_train_transition(config_to_predict_list, self.dynamic_oracle)
            new_config_to_predict_list = []
            for i in range(len(config_to_predict_list)):
                config = config_to_predict_list[i]
                best, shift_case = best_transition_list[i]
                self.train_transition_processing(config, best, shift_case)
                self.oracle.error_processing(False)
                if not config.is_end():
                    new_config_to_predict_list.append(config)
            config_to_predict_list = new_config_to_predict_list
            iter_num += 1
        self.oracle.error_processing(True)
        mloss = self.oracle.get_mloss()

        self.info_logger.info(f"Loss: {mloss / len(trainData)}")
        self.info_logger.info(f"Total Training Time: {time.time() - beg:.2g}s")
