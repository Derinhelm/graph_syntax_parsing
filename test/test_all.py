import pytest
import sys
sys.path.append('./src')
#sys.path.append('../src')

from project_logging import logger_creating
logger_creating() # TODO: переместить в тесты?

from utils import set_seeds
from run import run, create_options
from data import get_data

@pytest.mark.parametrize("context_embed_flag",
                         [True, False])
@pytest.mark.parametrize("mode",
                         ["mlp", "graph"])
def test_full_cycle(context_embed_flag, mode):
    import os
    os.system("rm -fr new_models; mkdir new_models") # TODO

    # really important to do this before anything else to make experiments reproducible
    set_seeds()

    train, val, test = get_data(context_embed=context_embed_flag,
        real_dataset=False, embed_pickle_using=True, colab=False)

    options = create_options(epochs=3)
    run(train, val, test, options=options, mode=mode)
