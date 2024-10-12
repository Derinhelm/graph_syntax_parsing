from configuration_graph import ConfigGraph
from constants import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP

class GraphConfigurationEmbedder:
    def __init__(self, device, mode, init_config):
        self.device = device
        self.mode = mode
        self.graph = ConfigGraph(init_config.sentence, init_config.word_embeds, device)
        self.config_embed = self._create_config_embed(init_config)

    def _create_config_embed(self, config):
        return self.graph.get_graph()

    def get_config_embed(self):
        r = self.config_embed
        print(f"get_config_embed:{r}")
        return r

    def apply_transition(self, transition, graph_info, config):
        apply_function_dict = { SHIFT: self.graph.apply_shift
                              , SWAP: self.graph.apply_swap
                              , LEFT_ARC: self.graph.apply_left_arc
                              , RIGHT_ARC: self.graph.apply_right_arc
                              }

        apply_function = apply_function_dict[transition]
        apply_function(*graph_info)
        self.config_embed = self._create_config_embed(config)
