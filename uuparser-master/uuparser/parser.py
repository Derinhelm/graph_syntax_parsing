from optparse import OptionParser, OptionGroup
from uuparser.options_manager import OptionsManager
import pickle, os, time, sys, copy, itertools, re, random

from loguru import logger
from shutil import copyfile

from uuparser import utils


def run(experiment,options):
    
    from uuparser.arc_hybrid import ArcHybridLSTM
    #logger.info('Working with a transition-based parser')


    paramsfile = os.path.join(experiment.outdir, options.params)

    logger.debug('Preparing vocab')
    vocab = utils.get_vocab(experiment.treebanks,"train")
    logger.debug('Finished collecting vocab')

    with open(paramsfile, 'wb') as paramsfp:
            logger.info(f'Saving params to {paramsfile}')
            pickle.dump((vocab, options), paramsfp)

            logger.debug('Initializing the model')
            parser = ArcHybridLSTM(vocab, options)


    dev_best = [options.epochs,-1.0] # best epoch, best score

    for epoch in range(options.first_epoch, options.epochs+1):
        # Training
        logger.info(f'Starting epoch {epoch} (training)')
        traindata = list(utils.read_conll_dir(experiment.treebanks, "train", options.max_sentences))
        parser.Train(traindata,options)
        logger.info(f'Finished epoch {epoch} (training)')

        model_file = os.path.join(experiment.outdir, options.model + str(epoch))
        parser.Save(model_file)

        # not all treebanks necessarily have dev data
        pred_treebanks = [treebank for treebank in experiment.treebanks if treebank.pred_dev]
        if pred_treebanks:
                for treebank in pred_treebanks:
                    treebank.outfilename = os.path.join(treebank.outdir, 'dev_epoch_' + str(epoch) + '.conllu')
                    logger.info(f"Predicting on dev data for {treebank.name}")
                pred = list(parser.Predict(pred_treebanks,"dev",options))
                utils.write_conll_multiling(pred,pred_treebanks)

                if options.pred_eval: # evaluate the prediction against gold data
                    mean_score = 0.0
                    for treebank in pred_treebanks:
                        score = utils.evaluate(treebank.dev_gold,treebank.outfilename,options.conllu)
                        logger.info(f"Dev score {score:.2f} at epoch {epoch:d} for {treebank.name}")
                        mean_score += score
                    if len(pred_treebanks) > 1: # multiling case
                        mean_score = mean_score/len(pred_treebanks)
                        logger.info(f"Mean dev score {mean_score:.2f} at epoch {epoch:d}")
                    if options.model_selection:
                        if mean_score > dev_best[1]:
                            dev_best = [epoch,mean_score] # update best dev score
                        # hack to printthe word "mean" if the dev score is an average
                        mean_string = "mean " if len(pred_treebanks) > 1 else ""
                        logger.info(f"Best {mean_string}dev score {dev_best[1]:.2f} at epoch {dev_best[0]:d}")


            # at the last epoch choose which model to copy to barchybrid.model
        if epoch == options.epochs:
            bestmodel_file = os.path.join(experiment.outdir,"barchybrid.model" + str(dev_best[0]))
            model_file = os.path.join(experiment.outdir,"barchybrid.model")
            logger.info(f"Copying {bestmodel_file} to {model_file}")
            copyfile(bestmodel_file,model_file)
            best_dev_file = os.path.join(experiment.outdir,"best_dev_epoch.txt")
            with open(best_dev_file, 'w') as fh:
                logger.info(f"Writing best scores to: {best_dev_file}")
                if len(experiment.treebanks) == 1:
                    fh.write(f"Best dev score {dev_best[1]} at epoch {dev_best[0]:d}\n")
                else:
                    fh.write(f"Best mean dev score {dev_best[1]} at epoch {dev_best[0]:d}\n")
            
            # evaluation for the epoche
        ts = time.time()

        pred = list(parser.Predict(experiment.treebanks,"test",stored_opt))
        utils.write_conll_multiling(pred,experiment.treebanks)

        te = time.time()

        if options.pred_eval:
            for treebank in experiment.treebanks:
                logger.debug(f"Evaluating on {treebank.name}")
                score = utils.evaluate(treebank.test_gold,treebank.outfilename,options.conllu)
                logger.info(f"Obtained LAS F1 score of {score:.2f} on {treebank.name}")

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
    parser.add_option("--params", metavar="FILE", default="params.pickle", help="Parameters file")
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

    om = OptionsManager(options)
    experiments = om.create_experiment_list(options) # list of namedtuples
    for experiment in experiments:
        run(experiment,options)

if __name__ == '__main__':
    main()