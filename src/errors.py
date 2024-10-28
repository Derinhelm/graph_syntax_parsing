from logging import getLogger
import time
import torch

from constants import LEFT_ARC, RIGHT_ARC
from metrics_logging import save_metric

class ErrorInfo:
    def __init__(self):
        self.train_info = self.create_train_info()

    def create_train_info(self):
        t_info = {}
        t_info["mloss"], t_info["eloss"], t_info["eerrors"], t_info["lerrors"], t_info["etotal"] \
            = 0.0, 0.0, 0, 0, 0
        t_info["errs"] = torch.tensor(0.0, requires_grad=True)
        t_info["sentence_ind"] = -1
        t_info["start"] = time.time()
        return t_info

    def error_append(self, best, bestValid, bestWrong, config):
        #labeled errors
        if best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            child = config.get_stack_last_element() # Last stack element
            if (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                self.train_info["lerrors"] += 1
                #attachment error
                if child.pred_parent_id != child.parent_id:
                    self.train_info["eerrors"] += 1
        
        if bestValid[2] < bestWrong[2] + 1.0:
            self.train_info["mloss"] += 1.0 + bestWrong[2] - bestValid[2]
            self.train_info["eloss"] += 1.0 + bestWrong[2] - bestValid[2]
            loss = bestWrong[3] - bestValid[3] # values in computational graph
            #print(f"loss:{loss}")
            self.train_info["errs"] = self.train_info["errs"] + loss
            #print(f"self.train_info['errs']:{self.train_info['errs']}", f"loss:{loss}")
            save_metric("loss", bestWrong[2] - bestValid[2])

        #??? when did this happen and why?
        if best[1] == 0 or best[1] == 2:
            self.train_info["etotal"] += 1

    def train_logging(self):
        loss_message = (
            f'Processing sentence number: {self.train_info["sentence_ind"]}'
            f' Loss: {self.train_info["eloss"] / self.train_info["etotal"]:.3f}'
            f' Errors: {self.train_info["eerrors"] / self.train_info["etotal"]:.3f}'
            f' Labeled Errors: {self.train_info["lerrors"] / self.train_info["etotal"]:.3f}'
            f' Time: {time.time() - self.train_info["start"]:.3f}s'
        )
        info_logger = getLogger('info_logger')
        info_logger.debug(loss_message)
        self.train_info["start"] = time.time() # TODO: зачем этот параметр ?
        self.train_info["eerrors"], self.train_info["eloss"], self.train_info["etotal"], self.train_info["lerrors"] = \
            0, 0.0, 0, 0 # TODO: Почему здесь зануляем?

    def get_mloss(self):
        return self.train_info['mloss']

    def change_sentence_number(self, sentence_ind):
        self.train_info["sentence_ind"] = sentence_ind

    def set_errs(self):
        self.train_info["errs"] = torch.tensor(0.0, requires_grad=True)
    
    def get_errs(self):
        return self.train_info["errs"]
