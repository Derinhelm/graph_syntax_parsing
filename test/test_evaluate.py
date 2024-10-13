import copy
import sys
sys.path.append('./src')
#sys.path.append('../src')

from project_logging import logger_creating
import torch
logger_creating() # TODO: переместить в тесты?

from data import get_data
from project_parser import Parser
from run import create_options
from utils import set_seeds, get_irels

def test_full_cycle():
    import os
    os.system("rm -fr new_models; mkdir new_models") # TODO

    # really important to do this before anything else to make experiments reproducible
    set_seeds()

    train, val, test = get_data(context_embed=False,
        real_dataset=False, embed_pickle_using=True, colab=False)

    options = create_options(epochs=3, learning_rate=0.1)
    irels = get_irels(train)
    parser = Parser(options, irels, 'mlp', 'depth')
    parser.oracle.net.net.train()
    old_net_state = copy.deepcopy(list(parser.oracle.net.net.parameters()))
    parser.Train(train)
    new_net_state = list(parser.oracle.net.net.parameters())
    assert len(old_net_state) == len(new_net_state)
    equal_list = [torch.equal(old_net_state[i], new_net_state[i])
                  for i in range(len(old_net_state))]
    assert not all(equal_list)
