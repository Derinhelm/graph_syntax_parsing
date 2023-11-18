import random
import time
import tqdm


from configuration import Configuration
from constants import SWAP
from oracle import Oracle
from project_logging import logging
from utils import ConllEntry



class Parser:
    def __init__(self, options, irels, embeds):
        self.dynamic_oracle = options["dynamic_oracle"]
        self.oracle = Oracle(options, irels, embeds)

    def Load(self, epoch):
        self.oracle.Load(epoch)

    def Save(self, epoch):
        self.oracle.Save(epoch)

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

        for iSentence, osentence in enumerate(pbar,1):
            config = Configuration(osentence, self.oracle.irels)
            max_swap = 2*len(osentence)
            reached_swap_for_i_sentence = False
            iSwap = 0

            while not config.is_end():
                best = self.oracle.create_test_transition(config, iSwap, max_swap)
                if iSwap == max_swap and not reached_swap_for_i_sentence:
                    reached_max_swap += 1
                    reached_swap_for_i_sentence = True
                    logging.debug(f"reached max swap in {reached_max_swap:d} \
                                  out of {iSentence:d} sentences")
                config.apply_transition(best)
                if best[1] == SWAP:
                    iSwap += 1

            #keep in memory the information we need, not all the vectors
            oconll_sentence = [entry for entry in osentence if isinstance(entry, ConllEntry)]
            oconll_sentence = oconll_sentence[1:] + [oconll_sentence[0]]
            conll_sentence = config.get_sentence()
            for tok_o, tok in zip(oconll_sentence, conll_sentence):
                tok_o.pred_relation = tok.pred_relation
                tok_o.pred_parent_id = tok.pred_parent_id
            yield osentence

    def train_sentence(self, sentence):
        config = Configuration(sentence, self.oracle.irels)

        while not config.is_end():
            best, shift_case = self.oracle.create_train_transition(config, self.dynamic_oracle)

            #updates for the dynamic oracle
            if self.dynamic_oracle: # TODO: проверить, что значит True/False (где dynamic/static)
                config.dynamic_oracle_updates(best, shift_case)

            config.apply_transition(best)
        return

    def Train(self, trainData):
        random.shuffle(trainData)
        # in certain cases the data will already have been shuffled after being read from file or while creating dev data
        logging.info(f"Length of training data: {len(trainData)}")

        beg = time.time()

        pbar = tqdm.tqdm(
            trainData, desc="Training", unit="sentences",
            mininterval=1.0, leave=False, disable=False,
        )

        for iSentence, sentence in enumerate(pbar,1):
            self.oracle.change_sentence_number(iSentence)
            if iSentence % 100 == 0:
                self.oracle.train_logging()

            self.train_sentence(sentence)
            self.oracle.error_processing(False)

        self.oracle.error_processing(True)

        mloss = self.oracle.get_mloss()

        logging.info(f"Loss: {mloss / iSentence}")
        logging.info(f"Total Training Time: {time.time() - beg:.2g}s")
