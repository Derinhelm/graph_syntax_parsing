from logging import getLogger
import time

from metrics_logging import save_metric
from project_parser import Parser
from utils import ConllEntry, get_irels

def evaluate_uas(sentence_descr):
    right_parent_tokens = 0
    conll_len = 0
    for token in sentence_descr:
        if isinstance(token, ConllEntry):
          conll_len += 1
          if token.pred_parent_id == token.parent_id:
              right_parent_tokens += 1
    uas = right_parent_tokens / conll_len
    return uas

def evaluate_uas_epoche(sentence_list):
    summ_uas = 0
    for sent in sentence_list:
        summ_uas += evaluate_uas(sent)
    return summ_uas / len(sentence_list)

p = None

def set_logging(logging_params):
    info_logger = getLogger('info_logger')
    time_logger = getLogger('time_logger')
    transition_logger = getLogger('transition_logger')
    if not "INFO" in logging_params and len(info_logger.handlers) > 0:
        info_logger.removeHandler(info_logger.handlers[0])
    if not "TIME" in logging_params and len(time_logger.handlers) > 0:
        time_logger.removeHandler(time_logger.handlers[0])
    if not "TRANSITION" in logging_params and len(transition_logger.handlers) > 0:
        transition_logger.removeHandler(transition_logger.handlers[0])
    extra_logging_params = set(logging_params) - {"INFO", "TIME", "TRANSITION"}
    if len(extra_logging_params):
        print(f"Wrong logging params:{extra_logging_params}.")

def create_options(hidden_dims=100, learning_rate=0.001,\
        dynamic_oracle=True, epochs=10, first_epoch=1, elems_in_batch=1,):
    options = {}
    options["hidden_dims"] = hidden_dims # MLP hidden layer dimensions
    options["learning_rate"] = learning_rate # Learning rate for neural network optimizer

    options["dynamic_oracle"] = dynamic_oracle

    options["epochs"] = epochs # Number of epochs
    options["first_epoch"] = first_epoch
    options["elems_in_batch"] = elems_in_batch
    return options

def run(traindata, valdata, testdata, options={}, mode="mlp",
        batch_mode="breadth", logging_params=[]):
    ts = time.time()

    info_logger = getLogger('info_logger')
    set_logging(logging_params)
    irels = get_irels(traindata)
    parser = Parser(options, irels, mode, batch_mode)
    global p
    p = parser

    dev_best = [options["epochs"],-1.0] # best epoch, best score

    for epoch in range(options["first_epoch"], options["epochs"] + 1):
        # Training
        info_logger.info(f'Starting epoch {epoch} (training)')
        parser.Train(traindata)
        info_logger.info(f'Finished epoch {epoch} (training)')

        parser.Save(epoch)

        info_logger.info(f"Predicting on dev data")
        dev_pred = list(parser.Predict(valdata))
        mean_dev_score = evaluate_uas_epoche(dev_pred)
        info_logger.info(f"Dev score {mean_dev_score:.2f} at epoch {epoch:d}")
        print(f"Dev score {mean_dev_score:.2f} at epoch {epoch:d}")

        if mean_dev_score > dev_best[1]:
            dev_best = [epoch,mean_dev_score] # update best dev score
        
        save_metric.epoch += 1

    info_logger.info(f"Loading best model from epoche{dev_best[0]:d}")
    # Loading best_models to parser.labeled_GNN and parser.unlabeled_GNN
    parser.Load(epoch)

    info_logger.info(f"Predicting on test data")

    test_pred = list(parser.Predict(testdata))
    mean_test_score = evaluate_uas_epoche(test_pred)

    info_logger.info(f"On test obtained UAS score of {mean_test_score:.2f}")
    print(f"On test obtained UAS score of {mean_test_score:.2f}")


    info_logger.debug('Finished predicting')
    total_time = time.time() - ts
    info_logger.info(f"Time of all program: {total_time:.2f}")
    print(f"Total time of the program: {total_time:.2f}")
    return save_metric.metric_dict
