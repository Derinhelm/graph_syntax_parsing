import torch
from torch_geometric.data import HeteroData
from jax.numpy import int32


from copy import deepcopy

from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP
from utils import ConllEntry, ParseForest

def create_stack_edges(stack):
    if len(stack) == 0:
        return torch.stack((torch.tensor([], dtype=torch.int32), \
                            torch.tensor([], dtype=torch.int32)), dim=0)
    stack_edges = []
    if len(stack) == 1:
        stack_edges.append((stack[0].id - 1, stack[0].id - 1)) # temporary solution
    else:
        for i in range(len(stack) - 1): # Represents every two consecutive stack nodes as an edge
            stack_edges.append((stack[i].id - 1, stack[i + 1].id - 1))
    stack_edges = tuple(zip(*stack_edges))
    stack_edges = [torch.tensor(stack_edges[0]), torch.tensor(stack_edges[1])]
    return torch.stack(stack_edges, dim=0)

def create_buffer_edges(buffer):
    if len(buffer) == 0 or len(buffer) == 1: # Last element is a technical root element.
        return torch.stack((torch.tensor([], dtype=torch.int32), \
                            torch.tensor([], dtype=torch.int32)), dim=0)
    buffer_edges = []
    if len(buffer) == 2: # Last element is a technical root element.
        buffer_edges.append((buffer[0].id - 1, buffer[0].id - 1)) # temporary solution
    else:
        for i in range(len(buffer) - 2): # Last element is a technical root element.
        # Represents every two consecutive buffer nodes as an edge
            buffer_edges.append((buffer[i].id - 1, buffer[i + 1].id - 1))
    buffer_edges = tuple(zip(*buffer_edges))
    buffer_edges = [torch.tensor(buffer_edges[0]), torch.tensor(buffer_edges[1])]
    return torch.stack(buffer_edges, dim=0)

def create_graph_edges(sentence):
    graph_edges = []
    for node in sentence:
        if node.pred_parent_id is not None and node.pred_parent_id != 0 \
            and node.pred_parent_id != -1:
            graph_edges.append((node.pred_parent_id - 1, node.id - 1))
    if len(graph_edges) == 0:
        return torch.stack((torch.tensor([], dtype=torch.int32), \
                            torch.tensor([], dtype=torch.int32)), dim=0)
    graph_edges = tuple(zip(*graph_edges))
    graph_edges = [torch.tensor(graph_edges[0]), torch.tensor(graph_edges[1])]
    return torch.stack(graph_edges, dim=0)

class Configuration:
    def __init__(self, sentence, irels):
        self.sentence = deepcopy(sentence)
        # ensures we are working with a clean copy of sentence and allows memory to be recycled each time round the loop
        self.sentence = [entry for entry in self.sentence if isinstance(entry, ConllEntry)]
        self.sentence = self.sentence[1:] + [self.sentence[0]]
        self.stack = ParseForest([])
        self.buffer = ParseForest(self.sentence)
        for root in self.sentence:
            root.relation = root.relation if root.relation in irels else 'runk'


    def config_to_graph(self, embeds):
        word_embeds = torch.empty((len(self.sentence), 312))
        for i in range(len(self.sentence) - 1): # Last element is a technical root element.
            word_embeds[i] = embeds[self.sentence[i].lemma]

        data = HeteroData()
        data['node']['x'] = word_embeds

        data[('node', 'graph', 'node')].edge_index = create_graph_edges(self.sentence)
        data[('node', 'stack', 'node')].edge_index = create_stack_edges(self.stack.roots)
        data[('node', 'buffer', 'node')].edge_index = create_buffer_edges(self.buffer.roots)
        return data

    def apply_transition(self, best):
        if best[1] == SHIFT:
            self.stack.roots.append(self.buffer.roots[0])
            del self.buffer.roots[0]

        elif best[1] == SWAP:
            child = self.stack.roots.pop()
            self.buffer.roots.insert(1,child)

        elif best[1] == LEFT_ARC:
            child = self.stack.roots.pop()
            parent = self.buffer.roots[0]

        elif best[1] == RIGHT_ARC:
            child = self.stack.roots.pop()
            parent = self.stack.roots[-1]

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

    def __str__(self):
        return "stack:" + str(self.stack) + "\n" + "buffer:" + str(self.buffer) + "\n"

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
