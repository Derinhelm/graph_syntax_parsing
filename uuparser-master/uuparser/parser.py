from optparse import OptionParser, OptionGroup
from uuparser.options_manager import OptionsManager
import pickle, os, time, sys, copy, itertools, re, random

from loguru import logger
from shutil import copyfile

from uuparser import utils

def evaluate_uas(sentence_descr):
    #sentence_descr is a list, in which elements 0, 1, 2 are auxiliary
    right_parent_tokens = 0
    for token in sentence_descr[3:]:
        if isinstance(token, ConllEntry): # TODO: изучить случаи, когда не ConllEntry - ошибка считывания?
          if token.pred_parent_id == token.parent_id:
              right_parent_tokens += 1
        #print("pred_parent:", token.pred_parent_id, "real_parent:", token.parent_id)
    uas = right_parent_tokens / (len(sentence_descr) - 3)
    return uas

def evaluate_uas_epoche(sentence_list):
    summ_uas = 0
    for sent in sentence_list:
        summ_uas += evaluate_uas(sent)
    return summ_uas / len(sentence_list)

def run(traindata, valdata, testdata, options):
    
    from uuparser.arc_hybrid import ArcHybridLSTM
    #logger.info('Working with a transition-based parser')

    irels = utils.get_irels(traindata)
    logger.debug('Initializing the model')
    parser = ArcHybridLSTM(irels, options)

    dev_best = [options.epochs,-1.0] # best epoch, best score

    for epoch in range(options.first_epoch, options.epochs+1):
        # Training
        logger.info(f'Starting epoch {epoch} (training)')
        parser.Train(traindata,options)
        logger.info(f'Finished epoch {epoch} (training)')

        parser.Save(epoch)

        logger.info(f"Predicting on dev data")
        dev_pred = list(parser.Predict(valdata,"dev",options))
        mean_dev_score = evaluate_uas_epoche(dev_pred)
        logger.info(f"Dev score {mean_dev_score:.2f} at epoch {epoch:d}")
        print(f"Dev score {mean_dev_score:.2f} at epoch {epoch:d}")

        if mean_dev_score > dev_best[1]:
            dev_best = [epoch,mean_dev_score] # update best dev score

    logger.info(f"Loading best model from epoche{dev_best[0]:d}")
    # Loading best_models to parser.labeled_GNN and parser.unlabeled_GNN
    parser.Load(epoch)

    logger.info(f"Predicting on test data")

    test_pred = list(parser.Predict(testdata,"test",options))
    mean_test_score = evaluate_uas_epoche(test_pred)

    logger.info(f"On test obtained UAS score of {mean_test_score:.2f}")
    print(f"On test obtained UAS score of {mean_test_score:.2f}")


    logger.debug('Finished predicting')



def setup_logging():
    logger.remove(0)  # Remove the default logger
    
    log_level = "INFO"
    log_fmt = (
            "[uuparser] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
    )
    logger.add(
            utils.TqdmCompatibleStream(sys.stderr),
            level=log_level,
            format=log_fmt,
            colorize=True,
    )
    
    #logger.add(
    #        options.logfile,
    #       level="DEBUG",
    #        colorize=False,
    #)     
    #group.add_option("--logfile", metavar="FILE",
    #   help="A file where the training information will be logger")


def main():
    #group = OptionGroup(parser, "Experiment options") parser удалила
    group.add_option("--epochs", type="int", metavar="INTEGER", default=30,
        help='Number of epochs')
    group.add_option("--max-sentences", type="int", metavar="INTEGER",
        help='Only train using n sentences per epoch', default=-1) # Надо max_sentences
    group.add_option("--first-epoch", type="int", metavar="INTEGER", default=1) # first_epoch

    group = OptionGroup(parser, "Transition-based parser options")
    group.add_option("--disable-oracle", action="store_false", dest="oracle", default=True,
        help='Use the static oracle instead of the dynamic oracle') # oracle
    group.add_option("--disable-head", action="store_false", dest="headFlag", default=True,
        help='Disable using the head of word vectors fed to the MLP') # headFlag
    group.add_option("--disable-rlmost", action="store_false", dest="rlMostFlag", default=True,
        help='Disable using leftmost and rightmost dependents of words fed to the MLP') # rlMostFlag
    group.add_option("--userl", action="store_true", dest="rlFlag", default=False)
    group.add_option("--k", type="int", metavar="INTEGER", default=3,
        help="Number of stack elements to feed to MLP")
    parser.add_option_group(group)

    group.add_option("--learning-rate", type="float", metavar="FLOAT",
        help="Learning rate for neural network optimizer", default=0.001)
    group.add_option("--mlp-hidden-dims", type="int", metavar="INTEGER",
        help="MLP hidden layer dimensions", default=100)
    group.add_option("--activation", help="Activation function in the MLP", default="tanh")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()

    setup_logging()

    # really important to do this before anything else to make experiments reproducible
    utils.set_seeds()

    train_dir = 'sample_data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu'
    val_dir = 'sample_data/UD_Russian-SynTagRus/ru_syntagrus-ud-val.conllu'
    test_dir = 'sample_data/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu'

    train = list(read_conll(train_dir, maxSize=options.max_sentences))
    val = list(read_conll(val_dir, maxSize=options.max_sentences))
    test = list(read_conll(test_dir, maxSize=options.max_sentences))
    run(train, val, test, options)

if __name__ == '__main__':
    main()