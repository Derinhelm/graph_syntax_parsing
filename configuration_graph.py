from logging import getLogger
import time
import torch
from torch_geometric.data import HeteroData

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP

def create_buffer_edges(buffer):
    buffer_edges = []
    for i in range(len(buffer) - 1):
        # Represents every two consecutive buffer nodes as an edge
        buffer_edges.append((buffer[i].id, buffer[i + 1].id))
    buffer_edges = tuple(zip(*buffer_edges))
    buffer_edges = [torch.tensor(buffer_edges[0]), torch.tensor(buffer_edges[1])]
    return torch.stack(buffer_edges, dim=0)

class ConfigGraph:
    def __init__(self, sentence, word_embeds, device):
        self.word_embeds = word_embeds
        self.sentence = sentence
        time_logger = getLogger('time_logger')
        ts = time.time()
        buffer = [w for w in sentence]
        self.data = HeteroData()
        self.data['node']['x'] = word_embeds

        self.data[('node', 'graph', 'node')].edge_index = torch.stack((torch.tensor([], dtype=torch.int32),
                                torch.tensor([], dtype=torch.int32)), dim=0)
        self.data[('node', 'stack', 'node')].edge_index = torch.stack((torch.tensor([], dtype=torch.int32),
                                torch.tensor([], dtype=torch.int32)), dim=0)
        self.data[('node', 'buffer', 'node')].edge_index = create_buffer_edges(buffer)
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

    def apply_shift(self, buf_0, buf_1, stack_last):
        ''' self.stack.roots.append(self.buffer.roots[0])
            del self.buffer.roots[0]'''

        self.remove_buffer_edge(buf_0, buf_1)
        if stack_last is not None:
            self.add_stack_edge(stack_last, buf_0)
        else:
            self.add_stack_edge(buf_0, None) # Create a technical cycle (for 1 token in stack)

    def apply_left_arc(self, parent, child, stack_new_last, rel_type):
        if stack_new_last is not None:
            self.remove_stack_edge(stack_new_last, child)
        else:
            self.remove_stack_edge(child, None)
        self.add_edge(('node', 'graph', 'node'), parent, child)

    def apply_right_arc(self, parent, child, rel_type):
        self.remove_stack_edge(parent, child)
        self.add_edge(('node', 'graph', 'node'), parent, child)

    def apply_swap(self, buf_0, buf_1, stack_new_last, child):
      '''
      child = self.stack.roots.pop()
      self.buffer.roots.insert(1,child)
      '''
      if stack_new_last is not None:
          self.remove_stack_edge(stack_new_last, child)
      else:
          self.remove_stack_edge(child, None)
      if buf_1 is not None:
          self.remove_buffer_edge(buf_0, buf_1)
          self.add_buffer_edge(buf_0, child)
          self.add_buffer_edge(child, buf_1)
      else:
          self.add_buffer_edge(buf_0, child)

    def get_graph(self):
        return self.data

    def get_dicts(self):
        return self.data.x_dict, self.data.edge_index_dict

    def remove_stack_edge(self, node_from_id, node_to_id):
      if node_to_id is None:
        if self.data[('node', 'stack', 'node')]['edge_index'].shape[1] == 1:
           self.remove_edge(('node', 'stack', 'node'), node_from_id, node_from_id)
        else:
          print("Error: Deleting last stack element from stack with len:", len(self.data[('node', 'stack', 'node')]))
      else:
        self.remove_edge(('node', 'stack', 'node'), node_from_id, node_to_id)
        if self.data[('node', 'stack', 'node')]['edge_index'].shape[1] == 0:
          # After deleting there is one element in the stack, create technical cycle.
          self.add_edge(('node', 'stack', 'node'), node_from_id, node_from_id)

    def remove_buffer_edge(self, node_from_id, node_to_id):
      if node_to_id is None:
        if self.data[('node', 'buffer', 'node')]['edge_index'].shape[1] == 1:
           self.remove_edge(('node', 'buffer', 'node'), node_from_id, node_from_id)
        else:
          print("Error: Deleting last buffer element from buffer with len:", len(self.data[('node', 'buffer', 'node')]))
      else:
        self.remove_edge(('node', 'buffer', 'node'), node_from_id, node_to_id)
        if self.data[('node', 'buffer', 'node')]['edge_index'].shape[1] == 0:
          # After deleting there is one element in the buffer, create technical cycle.
          self.add_edge(('node', 'buffer', 'node'), node_from_id, node_from_id)

    def add_buffer_edge(self, node_from_id, node_to_id):
      if node_to_id is None:
        if self.data[('node', 'buffer', 'node')]['edge_index'].shape[1] == 0:
          self.add_edge(('node', 'buffer', 'node'), node_from_id, node_from_id)
        else:
          print("Error: Adding first buffer element to buffer with len:", len(self.data[('node', 'buffer', 'node')]))
      else:
        if self.data[('node', 'buffer', 'node')]['edge_index'].shape[1] == 1 \
              and self.data[('node', 'buffer', 'node')]['edge_index'][0] == \
                    self.data[('node', 'buffer', 'node')]['edge_index'][1]:
            # Delete a technical cycle because of real edge creating
            self.remove_edge(('node', 'buffer', 'node'), node_from_id, node_from_id)
        self.add_edge(('node', 'buffer', 'node'), node_from_id, node_to_id)

    def remove_edge(self, edge_type, node_from_id, node_to_id):
      edge_storage = self.data[edge_type].edge_index
      existence_flag = False
      for i in range(len(edge_storage[0])):
        if edge_storage[0][i] == node_from_id and edge_storage[1][i] == node_to_id:
          existence_flag = True
          break
      if not existence_flag:
        print("No edge", node_from_id, node_to_id)
      else:
        self.data[edge_type].edge_index = \
          torch.cat((edge_storage[:,:i], edge_storage[:,i + 1:]), dim=1)

    def add_stack_edge(self, node_from_id, node_to_id):
      if node_to_id is None:
        if self.data[('node', 'stack', 'node')]['edge_index'].shape[1] == 0:
          self.add_edge(('node', 'stack', 'node'), node_from_id, node_from_id)
        else:
          print("Error: Adding first stack element to stack with len:", len(self.data[('node', 'stack', 'node')]))
      else:
        if self.data[('node', 'stack', 'node')]['edge_index'].shape[1] == 1 \
              and self.data[('node', 'stack', 'node')]['edge_index'][0] == \
                    self.data[('node', 'stack', 'node')]['edge_index'][1]:
            # Delete a technical cycle because of real edge creating
            self.remove_edge(('node', 'stack', 'node'), node_from_id, node_from_id)
        self.add_edge(('node', 'stack', 'node'), node_from_id, node_to_id)

    def add_edge(self, edge_type, node_from_id, node_to_id):
      edge_storage = self.data[edge_type].edge_index
      new_edge = torch.tensor([[node_from_id], [node_to_id]],
                              dtype=torch.int32).to(self.device)
      new_graph_index = torch.cat((edge_storage, new_edge), dim=1)
      self.data[edge_type].edge_index = new_graph_index

    def visualise(self):
      nx_graph = nx.MultiDiGraph()
      sentence_words = [self.sentence[-1].form] + [w.form for w in self.sentence[:-1]]
      sentence_words = [sentence_words[i] + "_" + str(i) for i in range(len(sentence_words))]
      for t, edg_indexes in self.data._edge_store_dict.items():
          edge_type = t[1]
          for i in range(len(edg_indexes['edge_index'][0])):
            node_from = int(edg_indexes['edge_index'][0][i])
            node_to = int(edg_indexes['edge_index'][1][i])
            nx_graph.add_edge(sentence_words[node_from], sentence_words[node_to], edge_type=edge_type)
      edge_types = nx.get_edge_attributes(nx_graph, 'edge_type')
      color = {'graph': 'blue', 'stack':'red', 'buffer':'green'}
      graph_edge_colors, additional_edge_colors = [], []
      graph_edges, additional_edges = [], []
      for edge in nx_graph.edges:
          if edge_types[edge] == 'graph':
            graph_edges.append(edge)
            graph_edge_colors.append(color[edge_types[edge]])
          else:
            additional_edges.append(edge)
            additional_edge_colors.append(color[edge_types[edge]])

      pos = graphviz_layout(nx_graph, prog='dot')

      nx.draw_networkx_edges(nx_graph, pos, edge_color=additional_edge_colors,
                connectionstyle='arc3, rad = -0.4', edgelist=additional_edges)
      nx.draw_networkx_edges(nx_graph, pos, edge_color=graph_edge_colors, edgelist=graph_edges)

      nx.draw_networkx_labels(nx_graph, pos)
      plt.show()

    def get_edges(self):
      def print_edges(edges):
        edges_from, edges_to = edges
        for i in range(len(edges_from)):
            print("    ", int(edges_from[i]), "->", int(edges_to[i]))
        if len(edges_from) == 0:
            print("    ", None)
      graph_edges = self.data._edge_store_dict[('node', 'graph', 'node')]['edge_index']
      stack_edges = self.data._edge_store_dict[('node', 'stack', 'node')]['edge_index']
      buffer_edges = self.data._edge_store_dict[('node', 'buffer', 'node')]['edge_index']

      print("graph:")
      print_edges(graph_edges)
      print("stack:")
      print_edges(stack_edges)
      print("buffer:")
      print_edges(buffer_edges)
