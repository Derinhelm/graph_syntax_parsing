from project_logging import logging
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

def run(traindata, valdata, testdata, embeds, options):

    irels = get_irels(traindata)
    logging.debug('Initializing the model')
    parser = Parser(options, irels, embeds)
    global p
    p = parser

    dev_best = [options["epochs"],-1.0] # best epoch, best score

    for epoch in range(options["first_epoch"], options["epochs"] + 1):
        # Training
        logging.info(f'Starting epoch {epoch} (training)')
        parser.Train(traindata)
        logging.info(f'Finished epoch {epoch} (training)')

        parser.Save(epoch)

        logging.info(f"Predicting on dev data")
        dev_pred = list(parser.Predict(valdata))
        mean_dev_score = evaluate_uas_epoche(dev_pred)
        logging.info(f"Dev score {mean_dev_score:.2f} at epoch {epoch:d}")
        print(f"Dev score {mean_dev_score:.2f} at epoch {epoch:d}")

        if mean_dev_score > dev_best[1]:
            dev_best = [epoch,mean_dev_score] # update best dev score

    logging.info(f"Loading best model from epoche{dev_best[0]:d}")
    # Loading best_models to parser.labeled_GNN and parser.unlabeled_GNN
    parser.Load(epoch)

    logging.info(f"Predicting on test data")

    test_pred = list(parser.Predict(testdata))
    mean_test_score = evaluate_uas_epoche(test_pred)

    logging.info(f"On test obtained UAS score of {mean_test_score:.2f}")
    print(f"On test obtained UAS score of {mean_test_score:.2f}")


    logging.debug('Finished predicting')