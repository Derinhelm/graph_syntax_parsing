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
    # Loading best_models to parser.labeled_MLP and parser.unlabeled_MLP
    parser.Load(epoch)

    logger.info(f"Predicting on test data")

    test_pred = list(parser.Predict(testdata,"test",options))
    mean_test_score = evaluate_uas_epoche(test_pred)

    logger.info(f"On test obtained UAS score of {mean_test_score:.2f}")
    print(f"On test obtained UAS score of {mean_test_score:.2f}")


    logger.debug('Finished predicting')



def setup_logging(options):
    logger.remove(0)  # Remove the default logger
    if options.verbose:
        log_level = "DEBUG"
        log_fmt = (
            "[uuparser] "
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            "<level>{message}</level>"
        )
    else:
        log_level = "INFO"
        log_fmt = (
            "[uuparser] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
        )
    if not options.quiet:
        logger.add(
            utils.TqdmCompatibleStream(sys.stderr),
            level=log_level,
            format=log_fmt,
            colorize=True,
        )
    
    if options.logfile:
        logger.add(
            options.logfile,
            level="DEBUG",
            colorize=False,
        )


def main():
    parser = OptionParser()
    parser.add_option("--outdir", metavar="PATH", help='Output directory')
    parser.add_option("--datadir", metavar="PATH",
        help="Input directory with UD files; obligatory if using --include")
    parser.add_option("--testdir", metavar="PATH",
        help="Input directory with UD test files")
    parser.add_option("--golddir", metavar="PATH",
        help="Directory with gold UD test files (default is same as testdir)")
    parser.add_option("--modeldir", metavar="PATH",
        help='Directory where models will be saved, defaults to same as --outdir if not specified')
    parser.add_option("--model", metavar="FILE", default="barchybrid.model",
        help="Load/Save model file")

    group = OptionGroup(parser, "Experiment options")
    group.add_option("--include", metavar="LIST", help="List of languages by ISO code to be run \
if using UD. If not specified need to specify trainfile at least. When used in combination with \
--multiling, trains a common parser for all languages. Otherwise, train monolingual parsers for \
each")
    group.add_option("--trainfile", metavar="FILE", help="Annotated CONLL(U) train file")
    group.add_option("--devfile", metavar="FILE", help="Annotated CONLL(U) dev file")
    group.add_option("--testfile", metavar="FILE", help="Annotated CONLL(U) test file")
    group.add_option("--epochs", type="int", metavar="INTEGER", default=30,
        help='Number of epochs')
    group.add_option("--predict", help='Parse', action="store_true", default=False)
    group.add_option("--multiling", action="store_true", default=False,
        help='Train a multilingual parser with language embeddings')
    group.add_option("--max-sentences", type="int", metavar="INTEGER",
        help='Only train using n sentences per epoch', default=-1)
    group.add_option("--create-dev", action="store_true", default=False,
        help='Create dev data if no dev file is provided')
    group.add_option("--min-train-sents", type="int", metavar="INTEGER", default=1000,
        help='Minimum number of training sentences required in order to create a dev file')
    group.add_option("--dev-percent", type="float", metavar="FLOAT", default=5,
        help='Percentage of training data to use as dev data')
    group.add_option("--disable-pred-dev", action="store_false", dest="pred_dev", default=True,
        help='Disable prediction on dev data after each epoch')
    group.add_option("--disable-pred-eval", action="store_false", dest="pred_eval", default=True,
        help='Disable evaluation of prediction on dev data')
    group.add_option("--disable-model-selection", action="store_false",
        help="Disable choosing of model from best/last epoch", dest="model_selection", default=True)
    #TODO: reenable this
    group.add_option("--first-epoch", type="int", metavar="INTEGER", default=1)
    group.add_option("--predict-all-epochs", help='Ensures outfiles contain epoch number from model file',
        action="store_true", default=False)
    group.add_option("--forced-tbank-emb", type="string", default=None)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Transition-based parser options")
    group.add_option("--disable-oracle", action="store_false", dest="oracle", default=True,
        help='Use the static oracle instead of the dynamic oracle')
    group.add_option("--disable-head", action="store_false", dest="headFlag", default=True,
        help='Disable using the head of word vectors fed to the MLP')
    group.add_option("--disable-rlmost", action="store_false", dest="rlMostFlag", default=True,
        help='Disable using leftmost and rightmost dependents of words fed to the MLP')
    group.add_option("--userl", action="store_true", dest="rlFlag", default=False)
    group.add_option("--k", type="int", metavar="INTEGER", default=3,
        help="Number of stack elements to feed to MLP")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Neural network options")
    group.add_option("--learning-rate", type="float", metavar="FLOAT",
        help="Learning rate for neural network optimizer", default=0.001)
    group.add_option("--char-emb-size", type="int", metavar="INTEGER",
        help="Character embedding dimensions", default=500)
    group.add_option("--tbank-emb-size", type="int", metavar="INTEGER",
        help="Treebank embedding dimensions", default=12)
    group.add_option("--mlp-hidden-dims", type="int", metavar="INTEGER",
        help="MLP hidden layer dimensions", default=100)
    group.add_option("--mlp-hidden2-dims", type="int", metavar="INTEGER",
        help="MLP second hidden layer dimensions", default=0)
    group.add_option("--ext-word-emb-file", metavar="FILE",
                     help="External word embeddings")
    group.add_option("--ext-char-emb-file", metavar="FILE",
                     help="External character embeddings")
    group.add_option("--ext-emb-dir", metavar="PATH", help='Directory containing external embeddings')
    group.add_option("--max-ext-emb", type="int", metavar="INTEGER",
        help='Maximum number of external embeddings to load', default=-1)
    group.add_option("--activation", help="Activation function in the MLP", default="tanh")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Logging options")
    group.add_option("--verbose", action="store_true",
        help="Display more information during training", default=False)
    group.add_option("--quiet", action="store_true",
        help="Do not display anything during training", default=False)
    group.add_option("--logfile", metavar="FILE",
        help="A file where the training information will be logger")

    group = OptionGroup(parser, "Debug options")
    group.add_option("--debug", action="store_true",
        help="Run parser in debug mode, with fewer sentences", default=False)
    group.add_option("--debug-train-sents", type="int", metavar="INTEGER",
        help="Number of training sentences in --debug mode", default=20)
    group.add_option("--debug-dev-sents", type="int", metavar="INTEGER",
        help="Number of dev sentences in --debug mode", default=20)
    group.add_option("--debug-test-sents", type="int", metavar="INTEGER",
        help="Number of test sentences in --debug mode", default=20)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Shared task options")
    group.add_option("--shared-task", action="store_true", default=False)

    (options, args) = parser.parse_args()

    setup_logging(options)

    # really important to do this before anything else to make experiments reproducible
    utils.set_seeds(options)

    om = OptionsManager(options) # TODO: Now OptionsManager is used only for checking within it.
    # TODO: Create generating path from options
    train_dir = 'sample_data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu'
    val_dir = 'sample_data/UD_Russian-SynTagRus/ru_syntagrus-ud-val.conllu'
    test_dir = 'sample_data/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu'

    train = list(read_conll(train_dir, maxSize=options.max_sentences))
    val = list(read_conll(val_dir, maxSize=options.max_sentences))
    test = list(read_conll(test_dir, maxSize=options.max_sentences))
    run(train, val, test, options)

if __name__ == '__main__':
    main()