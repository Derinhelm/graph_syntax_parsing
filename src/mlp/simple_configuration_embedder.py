import torch

from constants import EMBED_SIZE

class SimpleConfigurationEmbedder:
    def __init__(self, device, mode, init_config):
        self.device = device
        self.mode = mode
        self.config_embed = self._create_config_embed(init_config)

    def _create_config_embed(self, config):
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
        old_embed = self.config_embed
        self.config_embed = self._create_config_embed(config)
        del old_embed
        if str(self.device) == "cuda":
            torch.cuda.empty_cache()
