from logging import getLogger

import time
import torch

from copy import deepcopy
from itertools import chain

from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP, EMBED_SIZE
from configuration_embedder import ConfigurationEmbedder
from utils import ConllEntry, ParseForest, generate_root_token

class Configuration:
    def __init__(self, sentence, irels, device, mode):
        self.device = device
        self.mode = mode
        self.sentence = deepcopy(sentence)
        # sentence = [ConllEntry_root, ConllEntry_1, ConllEntry_2, ...]
        root_token = self.sentence[0]
        self.stack = ParseForest([])
        self.buffer = ParseForest(self.sentence[1:] + [root_token])
        for root in self.sentence:
            root.relation = root.relation if root.relation in irels else 'runk'
        self.word_embeds = torch.empty((len(self.sentence) + 1, EMBED_SIZE))
        self.word_embeds[0] = root_token.start_embed
        for i, token in enumerate(self.sentence):
            self.word_embeds[i + 1] = token.start_embed
        self.config_embed_creator = ConfigurationEmbedder(self.device, self.mode, self)

    def get_config_embed(self):
        return self.config_embed_creator.get_config_embed()

    def __str__(self):
        s = "Config.\nsentence: " + ", ".join(map(str, self.sentence)) + "\n"
        s += "stack: " + str(self.stack) + "\n"
        s += "buffer: " + str(self.buffer) + "\n"
        return s

    def apply_transition(self, best):
        if best[1] == SHIFT:
            buf_0 = self.buffer.roots[0].id
            buf_1 = self.buffer.roots[1].id
            stack_last = self.stack.roots[-1].id if self.stack.roots != [] else None

            self.stack.roots.append(self.buffer.roots[0])
            del self.buffer.roots[0]
            trans_info = (buf_0, buf_1, stack_last)

        elif best[1] == SWAP:
            child = self.stack.roots.pop()
            child_id = child.id
            stack_new_last_id = self.stack.roots[-1].id if len(self.stack.roots) != 0 else None
            buf_0_old_id = self.buffer.roots[0].id
            buf_1_old_id = self.buffer.roots[1].id if len(self.buffer.roots) > 1 else None
            self.buffer.roots.insert(1,child)
            trans_info = (buf_0_old_id, buf_1_old_id, stack_new_last_id, child_id)

        elif best[1] == LEFT_ARC:
            child = self.stack.roots.pop()
            stack_new_last_id = self.stack.roots[-1].id if len(self.stack.roots) != 0 else None
            parent = self.buffer.roots[0]
            trans_info = (parent.id, child.id, stack_new_last_id, child.pred_relation)

        elif best[1] == RIGHT_ARC:
            child = self.stack.roots.pop()
            parent = self.stack.roots[-1]
            trans_info = (parent.id, child.id, child.pred_relation)
        else:
            print(f"Wrong transition:{best[1]}")
            exit(1) # TODO

        if best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            #attach
            child.pred_parent_id = parent.id
            child.pred_relation = best[0]

        self.config_embed_creator.apply_transition(best[1], trans_info, self)

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
