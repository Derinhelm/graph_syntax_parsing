from itertools import chain
from operator import itemgetter
import random

from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP

class TrainScores:
    def __init__(self, scrs, uscrs, non_detach_scrs, non_detach_uscrs):
        self.scrs = scrs
        self.uscrs = uscrs
        self.non_detach_scrs = non_detach_scrs
        self.non_detach_uscrs = non_detach_uscrs

    def _calculate_left_scores(self, config, irels):
        left_arc_conditions = config.check_left_arc_conditions()
        if not left_arc_conditions:
            return [], []
        left_cost = config.calculate_left_cost()
        left_scores = [(rel, LEFT_ARC, self.scrs[2 + j * 2] + self.uscrs[2], 
                            self.non_detach_scrs[2 + j * 2] + self.non_detach_uscrs[2]) \
                    for j, rel in enumerate(irels)]
        if left_cost == 0:
            left_valid_scores = [(rel, trans, sc, non_detach_sc) for (rel, trans, sc, non_detach_sc) in left_scores \
                if rel == config.get_stack_last_element().relation]
            left_wrong_scores = [(rel, trans, sc, non_detach_sc) for (rel, trans, sc, non_detach_sc) in left_scores \
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
        right_scores = [ (rel, RIGHT_ARC, self.scrs[3 + j * 2] + self.uscrs[3],
                          self.non_detach_scrs[3 + j * 2] + self.non_detach_uscrs[3]) \
                    for j, rel in enumerate(irels) ]

        if right_cost == 0:
            right_valid_scores = [(rel, trans, sc, non_detach_sc) for (rel, trans, sc, non_detach_sc) in right_scores \
                if rel == config.get_stack_last_element().relation]
            right_wrong_scores = [(rel, trans, sc, non_detach_sc) for (rel, trans, sc, non_detach_sc) in right_scores \
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

        shift_scores = [ (None, SHIFT, self.scrs[0] + self.uscrs[0], self.non_detach_scrs[0] + self.non_detach_uscrs[0]) ]

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

        swap_scores = [(None, SWAP, self.scrs[1] + self.uscrs[1], self.non_detach_scrs[1] + self.non_detach_uscrs[1])]

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

        valid = chain(left_valid, right_valid, shift_valid, swap_valid)
        wrong = chain(left_wrong, right_wrong, shift_wrong, swap_wrong, [(None, 4, -float('inf'), -float('inf'))]) # TODO: -float('inf') isn`t in computational graph
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

    def create_best_transaction(self, config, dynamic_oracle, error_info, irels):
        valid, wrong, shift_case, swap_cost = self.create_valid_wrong(config, irels)
        best_valid = max(valid, key=itemgetter(2))
        best_wrong = max(wrong, key=itemgetter(2))
        best = self.choose_best(best_valid, best_wrong, swap_cost, dynamic_oracle)
        error_info.error_append(best, best_valid, best_wrong, config)
        return best, shift_case

class TestScores:
    def __init__(self, scrs, uscrs):
        self.scrs = scrs
        self.uscrs = uscrs

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
