class BatchCreator:
    def __init__(self, config_list):
        self.config_list = config_list

    def get_new_batch(self):
        cur_config = self.config_list[0]
        if isinstance(cur_config, tuple): # for Predict
            cur_config = cur_config[0]
        if cur_config.is_end():
            self.config_list.pop(0)
        if len(self.config_list) > 0:
            return [self.config_list[0]]
        return None
