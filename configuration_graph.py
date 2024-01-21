from logging import getLogger
import time
import torch
from torch_geometric.data import HeteroData

from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP

def create_stack_edges(stack):
    if len(stack) == 0:
        return torch.stack((torch.tensor([], dtype=torch.int32), \
                            torch.tensor([], dtype=torch.int32)), dim=0)
    stack_edges = []
    if len(stack) == 1:
        stack_edges.append((stack[0].id, stack[0].id)) # temporary solution
    else:
        for i in range(len(stack) - 1): # Represents every two consecutive stack nodes as an edge
            stack_edges.append((stack[i].id, stack[i + 1].id))
    stack_edges = tuple(zip(*stack_edges))
    stack_edges = [torch.tensor(stack_edges[0]), torch.tensor(stack_edges[1])]
    return torch.stack(stack_edges, dim=0)

def create_buffer_edges(buffer):
    if len(buffer) == 0: # Last element is a technical root element.
        return torch.stack((torch.tensor([], dtype=torch.int32), \
                            torch.tensor([], dtype=torch.int32)), dim=0)
    buffer_edges = []
    if len(buffer) == 1: # Last element is a technical root element.
        buffer_edges.append((buffer[0].id, buffer[0].id)) # temporary solution
    else:
        for i in range(len(buffer) - 1):
        # Represents every two consecutive buffer nodes as an edge
            buffer_edges.append((buffer[i].id, buffer[i + 1].id))
    buffer_edges = tuple(zip(*buffer_edges))
    buffer_edges = [torch.tensor(buffer_edges[0]), torch.tensor(buffer_edges[1])]
    return torch.stack(buffer_edges, dim=0)

def create_graph_edges(sentence):
    graph_edges = []
    for node in sentence:
        if node.pred_parent_id is not None:
            graph_edges.append((node.pred_parent_id, node.id))
    if len(graph_edges) == 0:
        return torch.stack((torch.tensor([], dtype=torch.int32), \
                            torch.tensor([], dtype=torch.int32)), dim=0)
    graph_edges = tuple(zip(*graph_edges))
    graph_edges = [torch.tensor(graph_edges[0]), torch.tensor(graph_edges[1])]
    return torch.stack(graph_edges, dim=0)

class ConfigGraph:
    def __init__(self, sentence, stack, buffer, word_embeds, device):
        self.word_embeds = word_embeds
        time_logger = getLogger('time_logger')
        ts = time.time()
        self.data = HeteroData()
        self.data['node']['x'] = word_embeds
        transition_logger = getLogger('transition_logger')
        transition_logger.info("sentence:" + ", ".join(map(str, sentence)))
        transition_logger.info("graph.data['node']['x']. len:" + str(len(self.data['node']['x'])) + ", " + str(self.data['node']['x']))

        self.data[('node', 'graph', 'node')].edge_index = create_graph_edges(sentence)
        self.data[('node', 'stack', 'node')].edge_index = create_stack_edges(stack.roots)
        self.data[('node', 'buffer', 'node')].edge_index = create_buffer_edges(buffer.roots)
        time_logger.info(f"Time of config graph creating: {time.time() - ts}")
        self.device = device
        self.data.to(self.device)

    def __str__(self):
        s = "Graph edges: "
        s += "    graph:" + str(self.data._edge_store_dict[('node', 'graph', 'node')]['edge_index']) + "\n"
        s += "    stack:" + str(self.data._edge_store_dict[('node', 'stack', 'node')]['edge_index']) + "\n"
        s += "    buffer:" + str(self.data._edge_store_dict[('node', 'buffer', 'node')]['edge_index']) + "\n"
        s += "    graph dicts: " + str(self.get_dicts())
        return s

    def apply_transition(self, transition, sentence, stack, buffer):
        self.data = HeteroData()
        self.data['node']['x'] = self.word_embeds

        self.data[('node', 'graph', 'node')].edge_index = create_graph_edges(sentence)
        self.data[('node', 'stack', 'node')].edge_index = create_stack_edges(stack.roots)
        self.data[('node', 'buffer', 'node')].edge_index = create_buffer_edges(buffer.roots)
        self.data.to(self.device)

    def get_dicts(self):
        return self.data.x_dict, self.data.edge_index_dict