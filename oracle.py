from itertools import chain
from logging import getLogger
from operator import itemgetter
import random
import time

from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP
from gnn import GNNNet

class Oracle:
    def __init__(self, options, irels, device):
        self.net = GNNNet(options, len(irels), device)
        self.irels = irels
        self.train_info = self.create_train_info()

    def create_train_info(self):
        t_info = {}
        t_info["mloss"], t_info["eloss"], t_info["eerrors"], t_info["lerrors"], t_info["etotal"] \
            = 0.0, 0.0, 0, 0, 0
        t_info["errs"] = []
        t_info["iSentence"] = -1
        t_info["start"] = time.time()
        return t_info

    def Load(self, epoch):
        self.net.Load(epoch)

    def Save(self, epoch):
        self.net.Save(epoch)


    def test_evaluate(self, config):
        """
        ret = [left arc,
               right arc
               shift]

        RET[i] = (rel, transition, score1) for shift, l_arc and r_arc
         shift = 2 (==> rel=None) ; l_arc = 0; r_acr = 1
        """
        scrs, uscrs = self.net.evaluate(config.graph)

        #transition conditions
        right_arc_conditions = config.check_right_arc_conditions()
        shift_conditions = config.check_shift_conditions()
        swap_conditions = config.check_swap_conditions()

        #(avoiding the multiple roots problem: disallow left-arc from root
        #if stack has more than one element
        left_arc_conditions = config.check_not_train_left_arc_conditions()

        s1,r1 = max(zip(scrs[2::2], self.irels))
        s2,r2 = max(zip(scrs[3::2], self.irels))
        s1 = s1 + uscrs[2]
        s2 = s2 + uscrs[3]
        ret = [ [ (r1, LEFT_ARC, s1) ] if left_arc_conditions else [],
                [ (r2, RIGHT_ARC, s2) ] if right_arc_conditions else [],
                [ (None, SHIFT, scrs[0] + uscrs[0]) ] if shift_conditions else [] ,
                [ (None, SWAP, scrs[1] + uscrs[1]) ] if swap_conditions else [] ]
        return ret

    def create_test_transition(self, config, iSwap, max_swap):
        scores = self.test_evaluate(config)
        best = max(chain(*(scores if iSwap < max_swap else scores[:3] )), key = itemgetter(2) )
        return best

    def calculate_left_scores(self, config, scrs, uscrs):
        left_arc_conditions = config.check_left_arc_conditions()
        if not left_arc_conditions:
            return [], []
        left_cost = config.calculate_left_cost()
        left_scores = [(rel, LEFT_ARC, scrs[2 + j * 2] + uscrs[2]) \
                    for j, rel in enumerate(self.irels)]
        if left_cost == 0:
            left_valid_scores = [(rel, trans, sc) for (rel, trans, sc) in left_scores \
                if rel == config.get_stack_last_element().relation]
            left_wrong_scores = [(rel, trans, sc) for (rel, trans, sc) in left_scores \
                if rel != config.get_stack_last_element().relation]

        else:
            left_valid_scores = []
            left_wrong_scores = left_scores
        return left_valid_scores, left_wrong_scores

    def calculate_right_scores(self, config, scrs, uscrs):
        right_arc_conditions = config.check_right_arc_conditions()
        if not right_arc_conditions:
            return [], []
        right_cost = config.calculate_right_cost()
        right_scores = [ (rel, RIGHT_ARC, scrs[3 + j * 2] + uscrs[3]) \
                    for j, rel in enumerate(self.irels) ]

        if right_cost == 0:
            right_valid_scores = [(rel, trans, sc) for (rel, trans, sc) in right_scores \
                if rel == config.get_stack_last_element().relation]
            right_wrong_scores = [(rel, trans, sc) for (rel, trans, sc) in right_scores \
                if rel != config.get_stack_last_element().relation]
        else:
            right_valid_scores = []
            right_wrong_scores = right_scores

        return right_valid_scores, right_wrong_scores


    def calculate_shift_scores(self, config, scrs, uscrs):
        shift_cost, shift_case = config.calculate_shift_cost()
        shift_conditions = config.check_shift_conditions()
        if not shift_conditions:
             return [], [], shift_case

        shift_scores = [ (None, SHIFT, scrs[0] + uscrs[0]) ]

        if shift_cost == 0:
            shift_valid_scores = shift_scores
            shift_wrong_scores = []
        else:
            shift_valid_scores = []
            shift_wrong_scores = shift_scores

        return shift_valid_scores, shift_wrong_scores, shift_case

    def calculate_swap_scores(self, config, scrs, uscrs):
        swap_conditions = config.check_swap_conditions()
        swap_cost = config.calculate_swap_cost()
        if not swap_conditions:
            return [], [], swap_cost

        swap_scores = [(None, SWAP, scrs[1] + uscrs[1])]

        if swap_cost == 0:
            swap_valid_scores = swap_scores
            swap_wrong_scores = []
        else:
            swap_valid_scores = []
            swap_wrong_scores = swap_scores

        return swap_valid_scores, swap_wrong_scores, swap_cost

    def error_append(self, best, bestValid, bestWrong, config):
        #labeled errors
        if best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            child = config.get_stack_last_element() # Last stack element
            if (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                self.train_info["lerrors"] += 1
                #attachment error
                if child.pred_parent_id != child.parent_id:
                    self.train_info["eerrors"] += 1

        if bestValid[2] < bestWrong[2] + 1.0:
            loss = bestWrong[2] - bestValid[2]
            self.train_info["mloss"] += 1.0 + bestWrong[2] - bestValid[2]
            self.train_info["eloss"] += 1.0 + bestWrong[2] - bestValid[2]
            self.train_info["errs"].append(loss)

        #??? when did this happen and why?
        if best[1] == 0 or best[1] == 2:
            self.train_info["etotal"] += 1

    def create_valid_wrong(self, config, scrs, uscrs):
        left_valid, left_wrong = self.calculate_left_scores(config, scrs, uscrs)
        right_valid, right_wrong = self.calculate_right_scores(config, scrs, uscrs)
        shift_valid, shift_wrong, shift_case = self.calculate_shift_scores(config, scrs, uscrs)
        swap_valid, swap_wrong, swap_cost = self.calculate_swap_scores(config, scrs, uscrs)

        valid = chain(left_valid, right_valid, shift_valid, swap_valid)
        wrong = chain(left_wrong, right_wrong, shift_wrong, swap_wrong, [(None, 4, -float('inf'))])
        # (None, 4, -float('inf')) is used to ensure that at least one element will be.
        return valid, wrong, shift_case, swap_cost

    def create_best(self, bestValid, bestWrong, swap_cost, config, dynamic_oracle):
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

    def create_train_transition(self, config, dynamic_oracle):
        time_logger = getLogger('time_logger')
        transition_logger = getLogger('transition_logger')

        ts = time.time()
        scrs, uscrs = self.net.evaluate(config.graph)
        time_logger.info(f"Time of net.evaluate: {time.time() - ts}")
        transition_logger.info("scrs:" + str(scrs))
        transition_logger.info("uscrs:" + str(uscrs))

        ts = time.time()
        valid, wrong, shift_case, swap_cost = self.create_valid_wrong(config, scrs, uscrs)

        best_valid = max(valid, key=itemgetter(2))
        best_wrong = max(wrong, key=itemgetter(2))
        transition_logger.info("best_valid:" + str(best_valid) + ", best_wrong:" + str(best_wrong))
        best = self.create_best(best_valid, best_wrong, swap_cost, config, dynamic_oracle)
        self.error_append(best, best_valid, best_wrong, config)
        time_logger.info(f"Time of create_best+: {time.time() - ts}")
        return best, shift_case

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

    def error_processing(self, is_final):
        # len(self.train_info["errs"]) > 50 is used because of footnote 8 in Eli's original paper
        if (is_final and len(self.train_info["errs"]) > 0) or (len(self.train_info["errs"]) > 50):
            self.net.error_processing(self.train_info["errs"])
            self.train_info["errs"] = []

    def get_mloss(self):
        return self.train_info['mloss']

    def change_sentence_number(self, iSentence):
        self.train_info["iSentence"] = iSentence

