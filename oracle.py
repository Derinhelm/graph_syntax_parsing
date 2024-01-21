from copy import deepcopy
from itertools import chain
from logging import getLogger
from operator import itemgetter
import random
import time

from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP
from gnn import GNNNet

class Scores:
    def __init__(self, scrs, uscrs):
        self.scrs = scrs
        self.uscrs = uscrs

    def _calculate_left_scores(self, config, irels):
        left_arc_conditions = config.check_left_arc_conditions()
        if not left_arc_conditions:
            return [], []
        left_cost = config.calculate_left_cost()
        left_scores = [(rel, LEFT_ARC, self.scrs[2 + j * 2] + self.uscrs[2]) \
                    for j, rel in enumerate(irels)]
        if left_cost == 0:
            left_valid_scores = [(rel, trans, sc) for (rel, trans, sc) in left_scores \
                if rel == config.get_stack_last_element().relation]
            left_wrong_scores = [(rel, trans, sc) for (rel, trans, sc) in left_scores \
                if rel != config.get_stack_last_element().relation]
        else:
            left_valid_scores = []
            left_wrong_scores = left_scores
        return left_valid_scores, left_wrong_scores

    def _calculate_right_scores(self, config, irels):
        right_arc_conditions = config.check_right_arc_conditions()
        if not right_arc_conditions:
            return [], []
        right_cost = config.calculate_right_cost()
        right_scores = [ (rel, RIGHT_ARC, self.scrs[3 + j * 2] + self.uscrs[3]) \
                    for j, rel in enumerate(irels) ]

        if right_cost == 0:
            right_valid_scores = [(rel, trans, sc) for (rel, trans, sc) in right_scores \
                if rel == config.get_stack_last_element().relation]
            right_wrong_scores = [(rel, trans, sc) for (rel, trans, sc) in right_scores \
                if rel != config.get_stack_last_element().relation]
        else:
            right_valid_scores = []
            right_wrong_scores = right_scores

        return right_valid_scores, right_wrong_scores

    def _calculate_shift_scores(self, config):
        shift_cost, shift_case = config.calculate_shift_cost()
        shift_conditions = config.check_shift_conditions()
        if not shift_conditions:
             return [], [], shift_case

        shift_scores = [ (None, SHIFT, self.scrs[0] + self.uscrs[0]) ]

        if shift_cost == 0:
            shift_valid_scores = shift_scores
            shift_wrong_scores = []
        else:
            shift_valid_scores = []
            shift_wrong_scores = shift_scores

        return shift_valid_scores, shift_wrong_scores, shift_case

    def _calculate_swap_scores(self, config):
        swap_conditions = config.check_swap_conditions()
        swap_cost = config.calculate_swap_cost()
        if not swap_conditions:
            return [], [], swap_cost

        swap_scores = [(None, SWAP, self.scrs[1] + self.uscrs[1])]

        if swap_cost == 0:
            swap_valid_scores = swap_scores
            swap_wrong_scores = []
        else:
            swap_valid_scores = []
            swap_wrong_scores = swap_scores

        return swap_valid_scores, swap_wrong_scores, swap_cost

    def create_valid_wrong(self, config, irels):
        left_valid, left_wrong = self._calculate_left_scores(config, irels)
        right_valid, right_wrong = self._calculate_right_scores(config, irels)
        shift_valid, shift_wrong, shift_case = self._calculate_shift_scores(config)
        swap_valid, swap_wrong, swap_cost = self._calculate_swap_scores(config)

        transition_logger = getLogger('transition_logger')
        transition_logger.info("left_valid:" + str(left_valid))
        transition_logger.info("left_wrong:" + str(left_wrong))
        transition_logger.info("right_valid:" + str(right_valid))
        transition_logger.info("right_wrong:" + str(right_wrong))
        transition_logger.info("shift_valid:" + str(shift_valid))
        transition_logger.info("shift_wrong:" + str(shift_wrong))
        transition_logger.info("swap_valid:" + str(swap_valid))
        transition_logger.info("swap_wrong:" + str(swap_wrong))
        valid = chain(left_valid, right_valid, shift_valid, swap_valid)
        wrong = chain(left_wrong, right_wrong, shift_wrong, swap_wrong, [(None, 4, -float('inf'))])
        # (None, 4, -float('inf')) is used to ensure that at least one element will be.
        return valid, wrong, shift_case, swap_cost

    def choose_best(self, bestValid, bestWrong, swap_cost, dynamic_oracle):
        #force swap
        if swap_cost == 0:
            best = bestValid
        else:
        #select a transition to follow
        # + aggresive exploration
        #1: might want to experiment with that parameter
            if bestWrong[1] == SWAP:
                best = bestValid
            else:
                best = bestValid if ( (not dynamic_oracle) or (bestValid[2] - bestWrong[2] > 1.0) \
                    or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong

        return best

    def transition_logging(self, valid, wrong, best_valid, best_wrong):
        transition_logger = getLogger('transition_logger')
        transition_logger.info("scrs:" + str(self.scrs))
        transition_logger.info("uscrs:" + str(self.uscrs))
        transition_logger.info("valid:" + str(list(valid)))
        transition_logger.info("wrong:" + str(list(wrong)))
        transition_logger.info("best_valid:" + str(best_valid) + ", best_wrong:" + str(best_wrong))

    def create_best_transaction(self, config, dynamic_oracle, error_info, irels):
        valid, wrong, shift_case, swap_cost = self.create_valid_wrong(config, irels)
        valid_copy = deepcopy(valid)
        wrong_copy = deepcopy(wrong)
        best_valid = max(valid, key=itemgetter(2))
        best_wrong = max(wrong, key=itemgetter(2))
        best = self.choose_best(best_valid, best_wrong, swap_cost, dynamic_oracle)
        error_info.error_append(best, best_valid, best_wrong, config)
        self.transition_logging(valid_copy, wrong_copy, best_valid, best_wrong)
        return best, shift_case

    def test_evaluate(self, config, irels):
        """
        ret = [left arc,
               right arc
               shift]

        RET[i] = (rel, transition, score1) for shift, l_arc and r_arc
         shift = 2 (==> rel=None) ; l_arc = 0; r_acr = 1
        """
        #transition conditions
        right_arc_conditions = config.check_right_arc_conditions()
        shift_conditions = config.check_shift_conditions()
        swap_conditions = config.check_swap_conditions()

        #(avoiding the multiple roots problem: disallow left-arc from root
        #if stack has more than one element
        left_arc_conditions = config.check_not_train_left_arc_conditions()

        s1,r1 = max(zip(self.scrs[2::2], irels))
        s2,r2 = max(zip(self.scrs[3::2], irels))
        s1 = s1 + self.uscrs[2]
        s2 = s2 + self.uscrs[3]
        ret = [ [ (r1, LEFT_ARC, s1) ] if left_arc_conditions else [],
                [ (r2, RIGHT_ARC, s2) ] if right_arc_conditions else [],
                [ (None, SHIFT, self.scrs[0] + self.uscrs[0]) ] if shift_conditions else [] ,
                [ (None, SWAP, self.scrs[1] + self.uscrs[1]) ] if swap_conditions else [] ]
        return ret

class ErrorInfo:
    def __init__(self):
        self.train_info = self.create_train_info()

    def create_train_info(self):
        t_info = {}
        t_info["mloss"], t_info["eloss"], t_info["eerrors"], t_info["lerrors"], t_info["etotal"] \
            = 0.0, 0.0, 0, 0, 0
        t_info["errs"] = []
        t_info["iSentence"] = -1
        t_info["start"] = time.time()
        return t_info

    def error_append(self, best, bestValid, bestWrong, config):
        #labeled errors
        if best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            child = config.get_stack_last_element() # Last stack element
            if (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                self.train_info["lerrors"] += 1
                #attachment error
                if child.pred_parent_id != child.parent_id:
                    self.train_info["eerrors"] += 1
        
        transition_logger = getLogger('transition_logger')
        if bestValid[2] < bestWrong[2] + 1.0:
            loss = bestWrong[2] - bestValid[2]
            self.train_info["mloss"] += 1.0 + bestWrong[2] - bestValid[2]
            self.train_info["eloss"] += 1.0 + bestWrong[2] - bestValid[2]
            self.train_info["errs"].append(loss)
            transition_logger.info("loss:" + str(loss))
        else:
            transition_logger.info("no loss")

        #??? when did this happen and why?
        if best[1] == 0 or best[1] == 2:
            self.train_info["etotal"] += 1

    def train_logging(self):
        loss_message = (
            f'Processing sentence number: {self.train_info["iSentence"]}'
            f' Loss: {self.train_info["eloss"] / self.train_info["etotal"]:.3f}'
            f' Errors: {self.train_info["eerrors"] / self.train_info["etotal"]:.3f}'
            f' Labeled Errors: {self.train_info["lerrors"] / self.train_info["etotal"]:.3f}'
            f' Time: {time.time() - self.train_info["start"]:.3f}s'
        )
        info_logger = getLogger('info_logger')
        info_logger.debug(loss_message)
        self.train_info["start"] = time.time() # TODO: зачем этот параметр ?
        self.train_info["eerrors"], self.train_info["eloss"], self.train_info["etotal"], self.train_info["lerrors"] = \
            0, 0.0, 0, 0 # TODO: Почему здесь зануляем?

    def get_mloss(self):
        return self.train_info['mloss']

    def change_sentence_number(self, iSentence):
        self.train_info["iSentence"] = iSentence

    def processing_check(self, is_final):
        # len(self.train_info["errs"]) > 50 is used because of footnote 8 in Eli's original paper
        return (is_final and len(self.train_info["errs"]) > 0) or (len(self.train_info["errs"]) > 50)

    def set_errs(self):
        self.train_info["errs"] = []
    
    def get_errs(self):
        return self.train_info["errs"]

class Oracle:
    def __init__(self, options, irels, device):
        self.net = GNNNet(options, len(irels), device)
        self.irels = irels
        self.error_info = ErrorInfo()

    def create_test_transition(self, config, iSwap, max_swap):
        scrs, uscrs = self.net.evaluate(config.graph.get_dicts())
        scores_info = Scores(scrs, uscrs)
        scores = scores_info.test_evaluate(config, self.irels)
        best = max(chain(*(scores if iSwap < max_swap else scores[:3] )), key = itemgetter(2) )
        return best

    def create_train_transition(self, config, dynamic_oracle):
        time_logger = getLogger('time_logger')

        ts = time.time()
        scrs, uscrs = self.net.evaluate(config.graph.get_dicts())
        time_logger.info(f"Time of net.evaluate: {time.time() - ts}")

        ts = time.time()
        scores_info = Scores(scrs, uscrs)
        best, shift_case = scores_info.create_best_transaction(config, dynamic_oracle, self.error_info, self.irels)
        time_logger.info(f"Time of create_best+: {time.time() - ts}")
        return best, shift_case

    def error_processing(self, is_final):
        if self.error_info.processing_check(is_final):
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

