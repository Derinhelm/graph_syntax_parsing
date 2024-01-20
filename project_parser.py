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

    def Load(self, epoch):
        self.oracle.Load(epoch)

    def Save(self, epoch):
        self.oracle.Save(epoch)

    def test_transaction_processing(self, config, best, max_swap, iSentence, \
                                    reached_max_swap, reached_swap_for_i_sentence, iSwap):
        info_logger = getLogger('info_logger')
        if iSwap == max_swap and not reached_swap_for_i_sentence:
            reached_max_swap += 1
            reached_swap_for_i_sentence = True
            info_logger.debug(f"reached max swap in {reached_max_swap:d} \
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
        pbar = tqdm.tqdm(
            data,
            desc="Parsing",
            unit="sentences",
            mininterval=1.0,
            leave=False,
            disable=False,
        )
        res_config_list = []
        for iSentence, osentence in enumerate(pbar,1):
            config = Configuration(osentence, self.oracle.irels, self.embeds, self.device)
            max_swap = 2*len(osentence)
            reached_swap_for_i_sentence = False
            iSwap = 0

            while not config.is_end():
                best = self.oracle.create_test_transition(config, iSwap, max_swap)
                reached_max_swap, reached_swap_for_i_sentence, iSwap = \
                        self.test_transaction_processing(config, best, max_swap, iSentence,\
                                                 reached_max_swap, reached_swap_for_i_sentence, iSwap)

            res_config_list.append(config)
            print("res_config_list:", res_config_list)

        for iSentence, osentence in enumerate(data,1):
            config = res_config_list[iSentence - 1]
            res_osentence = self.create_conll_res(osentence, config)
            yield res_osentence

    def train_transaction_processing(self, config, best, shift_case):
        time_logger = getLogger('time_logger')

        #updates for the dynamic oracle
        if self.dynamic_oracle:
            ts = time.time()
            config.dynamic_oracle_updates(best, shift_case)
            time_logger.info(f"Time of dynamic_oracle_updates: {time.time() - ts}")

        ts = time.time()
        config.apply_transition(best)
        time_logger.info(f"Time of apply_transition: {time.time() - ts}")
        return

    def Train(self, trainData):
        random.shuffle(trainData)
        info_logger = getLogger('info_logger')
        time_logger = getLogger('time_logger')
        # in certain cases the data will already have been shuffled after being read from file or while creating dev data
        info_logger.info(f"Length of training data: {len(trainData)}")

        beg = time.time()

        pbar = tqdm.tqdm(
            trainData, desc="Training", unit="sentences",
            mininterval=1.0, leave=False, disable=False,
        )
        time_logger = getLogger('time_logger')
        transition_logger = getLogger('transition_logger')
        config_list = []
        for sentence in trainData:
            config = Configuration(sentence, self.oracle.irels, self.embeds, self.device)
            config_list.append(config)
        while len(config_list) != 0:
            print("config_list:", config_list)
            new_config_list = []
            for config in config_list:
                transition_logger.info("--------------------")
                transition_logger.info(str(config))
                ts = time.time()
                best, shift_case = self.oracle.create_train_transition(config, self.dynamic_oracle)
                time_logger.info(f"Time of create_train_transition: {time.time() - ts}")
                transition_logger.info("best transition:" + str(best))
                self.train_transaction_processing(config, best, shift_case)
                ts = time.time()
                self.oracle.error_processing(False)
                time_logger.info(f"Time of error_processing: {time.time() - ts}")
                if not config.is_end():
                    new_config_list.append(config)
            config_list = new_config_list

        ts = time.time()
        self.oracle.error_processing(True)
        time_logger.info(f"Time of error_processing: {time.time() - ts}")

        mloss = self.oracle.get_mloss()

        info_logger.info(f"Loss: {mloss / len(trainData)}")
        info_logger.info(f"Total Training Time: {time.time() - beg:.2g}s")
