from logging import getLogger
import time

from project_parser import Parser
from utils import ConllEntry, get_irels

def evaluate_uas(sentence_descr):
    #sentence_descr is a list, in which elements 0, 1, 2 are auxiliary
    right_parent_tokens = 0
    for token in sentence_descr[3:]:
        if isinstance(token, ConllEntry): 
          # TODO: изучить случаи, когда не ConllEntry - ошибка считывания?
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

p = None

def run(traindata, valdata, testdata, embeds, hidden_dims=100, learning_rate=0.001,\
        dynamic_oracle=True, epochs=10, first_epoch=1, info_logging=True, \
        time_logging=True, transition_logging=True, elems_in_batch=1):
    ts = time.time()
    options = {}
    options["hidden_dims"] = hidden_dims # MLP hidden layer dimensions
    options["learning_rate"] = learning_rate # Learning rate for neural network optimizer

    options["dynamic_oracle"] = dynamic_oracle

    options["epochs"] = epochs # Number of epochs
    options["first_epoch"] = first_epoch
    options["elems_in_batch"] = elems_in_batch

    info_logger = getLogger('info_logger')
    time_logger = getLogger('time_logger')
    transition_logger = getLogger('transition_logger')
    if not info_logging and len(info_logger.handlers) > 0:
        info_logger.removeHandler(info_logger.handlers[0])
    if not time_logging and len(time_logger.handlers) > 0:
        time_logger.removeHandler(time_logger.handlers[0])
    if not transition_logging and len(transition_logger.handlers) > 0:
        transition_logger.removeHandler(transition_logger.handlers[0])
    irels = get_irels(traindata)
    info_logger.debug('Initializing the model')
    parser = Parser(options, irels, embeds)
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
    time_logger.info(f"Time of all program: {total_time:.2f}")
    print(f"Total time of the program: {total_time:.2f}")