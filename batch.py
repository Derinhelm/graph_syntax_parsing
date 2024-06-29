from collections import deque

class BatchCreator:
    def __init__(self, first_config_list, batch_size):
        self.cur_config_list = deque(
            [first_config_list[i:i + batch_size]
             for i in range(0, len(first_config_list), batch_size)])
        self.batch_size = batch_size
        self.unprocessed_lists = deque()

    def _create_new_cur_list(self, config_list):
        return deque(
            [config_list[i:i + self.batch_size]
             for i in range(0, len(config_list), self.batch_size)])

    def get_new_batch(self):
        if len(self.cur_config_list) != 0:
            return self.cur_config_list.popleft()
        if len(self.unprocessed_lists) != 0:
            new_list = self.unprocessed_lists.popleft()
            self.cur_config_list = \
                self._create_new_cur_list(new_list)
            return self.cur_config_list.popleft()
        return None

    def add_new_config_list(self, new_config_list):
        if len(new_config_list):
            self.unprocessed_lists.append(new_config_list)
