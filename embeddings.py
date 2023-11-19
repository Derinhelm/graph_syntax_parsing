import time
from transformers import AutoTokenizer, BertModel
import pickle

from project_logging import logging

def get_embed(tokenizer, model, word): # TODO: переписать или убрать!
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state[0][0]
    return last_hidden_states.detach().cpu()

def create_embeds(all_words):
    embeds = {}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    logging.debug('Creating embeddings')
    ts = time.time()

    for word in all_words:
        embeds[word] = get_embed(tokenizer, model, word)
    logging.debug(f'{len(embeds)} embeddings were created')
    te = time.time()
    logging.info(f'Time of embedding creation: {te-ts:.2g}s')
    return embeds

def load_embeds(embed_pickle):
    ts = time.time()
    embeds = []
    with open(embed_pickle, 'rb') as f:
        embeds = pickle.load(f)
    te = time.time()
    logging.info(f'Time of embedding downloading: {te-ts:.2g}s')
    return embeds