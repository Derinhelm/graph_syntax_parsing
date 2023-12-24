from logging import getLogger
import time
import torch
from transformers import AutoTokenizer, AutoModel
import pickle

def get_embed(tokenizer, model, word):
    inputs = tokenizer(word, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})

    last_hidden_states = outputs.last_hidden_state[0][0]
    return last_hidden_states.detach().cpu()

def create_embeds(all_words):
    info_logger = getLogger('info_logger')
    embeds = {}
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.cuda()  # uncomment it if you have a GPU

    info_logger.debug(f'Creating {len(all_words)} embeddings')
    ts = time.time()
    for word in all_words:
        embeds[word] = get_embed(tokenizer, model, word)
    info_logger.debug(f'{len(embeds)} embeddings were created')
    te = time.time()
    info_logger.info(f'Time of embedding creation: {te-ts:.2g}s')
    return embeds

def load_embeds(embed_pickle):
    time_logger = getLogger('time_logger')
    ts = time.time()
    embeds = []
    with open(embed_pickle, 'rb') as f:
        embeds = pickle.load(f)
    te = time.time()
    time_logger.info(f'Time of embedding downloading: {te-ts:.2g}s')
    return embeds