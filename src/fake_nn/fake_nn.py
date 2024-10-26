import torch

class FakeBlock:
    def train(self):
        pass

    def eval(self):
        pass

class FakeNet:
    def __init__(self, options, out_irels_dims, device):
        self.unlabeled_res_size = 4 # for a single element in batch
        self.labeled_res_size = 2 * out_irels_dims + 2 # for a single element in batch
        self.net = FakeBlock()

    def Load(self, epoch):
        x = 0

    def Save(self, epoch):
        x = 0

    def evaluate(self, batch_embeds):
        batch_elem_count = len(batch_embeds)
        return list(torch.ones(batch_elem_count, self.unlabeled_res_size + self.labeled_res_size)),\
            list(torch.ones(batch_elem_count, self.unlabeled_res_size + self.labeled_res_size))

    def get_scrs_uscrs(self, all_scrs):
        uscrs = all_scrs[:self.unlabeled_res_size]
        scrs = all_scrs[self.unlabeled_res_size:]
        return scrs, uscrs

    def error_processing(self, errs):
        x = 0
