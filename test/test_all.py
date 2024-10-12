import sys
sys.path.append('..')

from project_logging import logger_creating
logger_creating()

from utils import set_seeds
from run import run
from data import get_data

def test_full_cycle():
    import os
    os.system("mkdir new_models") # TODO

    # really important to do this before anything else to make experiments reproducible
    set_seeds()

    train, val, test, embeds = get_data(real_dataset=False, embed_pickle_using=True, colab=False)

    run(train, val, test, embeds, epochs=3)
