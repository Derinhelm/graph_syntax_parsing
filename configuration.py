from logging import getLogger

import time
import torch

from copy import deepcopy

from configuration_graph import ConfigGraph
from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP
from utils import ConllEntry, ParseForest

class Configuration:
    def __init__(self, sentence, irels, embeds, device):
        self.sentence = deepcopy(sentence)
        # ensures we are working with a clean copy of sentence and allows memory to be recycled each time round the loop
        self.sentence = [entry for entry in self.sentence if isinstance(entry, ConllEntry)]
        self.sentence = self.sentence[1:] + [self.sentence[0]]
        self.stack = ParseForest([])
        self.buffer = ParseForest(self.sentence)
        for root in self.sentence:
            root.relation = root.relation if root.relation in irels else 'runk'
        embed_size = 312 # for tiny-bert
        self.word_embeds = torch.empty((len(self.sentence), embed_size))
        self.word_embeds[0] = torch.zeros(embed_size) # TODO: temporary solution for root element
        for i in range(len(self.sentence) - 1): # Last element is a technical root element.
            self.word_embeds[i + 1] = embeds[self.sentence[i].lemma] # Word number id starts from 1 in the graph.

        self.graph = ConfigGraph(self.sentence, self.word_embeds, device)

    def get_config_embed(self, device, mode="graph"):
        if mode == "graph":
            return self.graph.get_graph()
        else:
            buffer_id = self.buffer.roots[0].id
            e = self.word_embeds[buffer_id].to(device)
            return e

    def __str__(self):
        s = "Config.\nsentence: " + ", ".join(map(str, self.sentence)) + "\n"
        s += "stack: " + str(self.stack) + "\n"
        s += "buffer: " + str(self.buffer) + "\n"
        s += "graph:" + str(self.graph)
        return s

    def apply_transition(self, best):
        #time_logger = getLogger('time_logger')
        #ts = time.time()
        if best[1] == SHIFT:
            buf_0 = self.buffer.roots[0].id
            buf_1 = self.buffer.roots[1].id
            stack_last = self.stack.roots[-1].id if self.stack.roots != [] else None

            self.stack.roots.append(self.buffer.roots[0])
            del self.buffer.roots[0]
            #ts = time.time()
            self.graph.apply_shift(buf_0, buf_1, stack_last)
            #time_logger.info(f"Time of graph apply_transition: {time.time() - ts}")

        elif best[1] == SWAP:
            child = self.stack.roots.pop()
            child_id = child.id
            stack_new_last_id = self.stack.roots[-1].id if len(self.stack.roots) != 0 else None
            buf_0_old_id = self.buffer.roots[0].id
            buf_1_old_id = self.buffer.roots[1].id if len(self.buffer.roots) > 1 else None
            self.buffer.roots.insert(1,child)
            #ts = time.time()
            self.graph.apply_swap(buf_0_old_id, buf_1_old_id, stack_new_last_id, child_id)
            #time_logger.info(f"Time of graph apply_transition: {time.time() - ts}")


        elif best[1] == LEFT_ARC:
            child = self.stack.roots.pop()
            stack_new_last_id = self.stack.roots[-1].id if len(self.stack.roots) != 0 else None
            parent = self.buffer.roots[0]
            #ts = time.time()
            self.graph.apply_left_arc(parent.id, child.id, stack_new_last_id, child.pred_relation)
            #time_logger.info(f"Time of graph apply_transition: {time.time() - ts}")

        elif best[1] == RIGHT_ARC:
            child = self.stack.roots.pop()
            parent = self.stack.roots[-1]
            #ts = time.time()
            self.graph.apply_right_arc(parent.id, child.id, child.pred_relation)
            #time_logger.info(f"Time of graph apply_transition: {time.time() - ts}")

        if best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            #attach
            child.pred_parent_id = parent.id
            child.pred_relation = best[0]

    def get_stack_ids(self):
        return [sitem.id for sitem in self.stack.roots]

    def is_stack_not_empty(self):
        return len(self.stack) > 0

    def get_stack_last_element(self):
        return self.stack.roots[-1] # Last stack element

    def get_stack_penultimate_element(self):
        return self.stack.roots[-2] # Penultimate stack element

    def get_buffer_head(self):
        return self.buffer.roots[0] # Head buffer element

    def get_buffer_tail(self):
        return self.buffer.roots[1:] if len(self.buffer) > 1 else [] # Tail of buffer

    def get_sentence(self):
        return self.sentence

    def is_end(self):
        return len(self.buffer) == 1 and len(self.stack) == 0

    def check_left_arc_conditions(self):
        return len(self.stack) > 0

    def check_not_train_left_arc_conditions(self):
            #(avoiding the multiple roots problem: disallow left-arc from root
            #if stack has more than one element
        return self.check_left_arc_conditions() and \
            not (self.buffer.roots[0].id == 0 and len(self.stack) > 1)

    def check_right_arc_conditions(self):
        return len(self.stack) > 1

    def check_shift_conditions(self):
        return self.buffer.roots[0].id != 0

    def check_swap_conditions(self):
        return len(self.stack) > 0 and self.stack.roots[-1].id < self.buffer.roots[0].id

    def calculate_left_cost(self):
        if not self.check_left_arc_conditions():
            return 1

        s0 = self.get_stack_last_element() # Last stack element
        b = self.get_buffer_head() # Head buffer element
        left_cost = len(s0.rdeps) + int(s0.parent_id != b.id and s0.id in s0.parent_entry.rdeps)

        if self.check_swap_conditions() and s0.projective_order > b.projective_order:
            left_cost = 1
        return left_cost

    def calculate_right_cost(self):
        if not self.check_right_arc_conditions():
            return 1

        s1 = self.get_stack_penultimate_element() # Penultimate stack element
        s0 = self.get_stack_last_element() # Last stack element
        b = self.get_buffer_head() # Head buffer element

        right_cost = len(s0.rdeps) + int(s0.parent_id != s1.id \
                                         and s0.id in s0.parent_entry.rdeps)

        if self.check_swap_conditions() and s0.projective_order > b.projective_order:
            right_cost = 1

        return right_cost

    def calculate_shift_cost(self):
        if not self.check_shift_conditions():
            shift_cost = 1
            shift_case = 0
            return shift_cost, shift_case

        b = self.get_buffer_head() # Head buffer element
        beta = self.get_buffer_tail() # Tail (list) of buffer

        if len([item for item in beta if item.projective_order < b.projective_order and \
                 item.id > b.id ])> 0:
            shift_cost = 0
            shift_case = 1
        else:
            stack_ids = self.get_stack_ids()
            shift_cost = len([d for d in b.rdeps if d in stack_ids]) + \
                int(self.is_stack_not_empty() and b.parent_id in stack_ids[:-1] \
                    and b.id in b.parent_entry.rdeps)
            shift_case = 2

        if self.check_swap_conditions():
            s0 = self.get_stack_last_element() # Last stack element
            if s0.projective_order > b.projective_order:
                shift_cost = 1

        return shift_cost, shift_case

    def calculate_swap_cost(self):
        if not self.check_swap_conditions():
            return 1

        s0 = self.get_stack_last_element() # Last stack element
        b = self.get_buffer_head() # Head buffer element

        if s0.projective_order > b.projective_order:
            swap_cost = 0
        else:
            swap_cost = 1

        return swap_cost

    def dynamic_oracle_updates(self, best, shift_case):
        stack_ids = self.get_stack_ids()
        if best[1] == SHIFT:
            if shift_case == 2:
                b = self.get_buffer_head() # Head buffer element
                if b.parent_entry.id in stack_ids[:-1] and b.id in b.parent_entry.rdeps:
                    b.parent_entry.rdeps.remove(b.id)
                blocked_deps = [d for d in b.rdeps if d in stack_ids]
                for d in blocked_deps:
                    b.rdeps.remove(d)

        elif best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            s0 = self.get_stack_last_element() # Last stack element
            s0.rdeps = []
            if s0.id in s0.parent_entry.rdeps:
                s0.parent_entry.rdeps.remove(s0.id)
