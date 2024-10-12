import torch

from configuration_graph import ConfigGraph
from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP, EMBED_SIZE

class ConfigurationEmbedder:
    def __init__(self, device, mode, init_config):
        self.device = device
        self.mode = mode
        self.graph = ConfigGraph(init_config.sentence, init_config.word_embeds, device)
        self.config_embed = self._create_config_embed(init_config)


    def _create_config_embed(self, config):
        if self.mode == "graph":
            return self.graph.get_graph()
        else:
            top_stack = [config.word_embeds[token.id] for token in config.stack.roots[-2:]]
            for _ in range(2 - len(top_stack)): # if stack doesn`t consist 2 tokens
                top_stack.append(torch.zeros(EMBED_SIZE))
            buffer_id = config.buffer.roots[0].id
            top_buffer = [config.word_embeds[buffer_id]] # buffer len should be more than 0
            embed = torch.cat(top_stack + top_buffer)
            embed = embed.to(self.device)
            return embed

    def get_config_embed(self):
        return self.config_embed

    def apply_transition(self, transition, graph_info, config):
        apply_function_dict = { SHIFT: self.graph.apply_shift
                              , SWAP: self.graph.apply_swap
                              , LEFT_ARC: self.graph.apply_left_arc
                              , RIGHT_ARC: self.graph.apply_right_arc
                              }
        
        apply_function = apply_function_dict[transition]
        apply_function(*graph_info)
        self.config_embed = self._create_config_embed(config)
