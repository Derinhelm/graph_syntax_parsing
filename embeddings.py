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

class TinyBertEmbeddingCreator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
        self.model.cuda()

    def create_first_bert_embeddings_sent(self, sent):
        inputs_ids_words = []
        word_start_position = []
        prev_word_finish = 0

        for word in sent:
            inputs = self.tokenizer(word, padding=True, truncation=True, return_tensors="pt")
            clear_inputs = inputs['input_ids'][0][1:-1] # without '[CLS]' and '[SEP]'
            inputs_ids_words.append(clear_inputs)
            word_start_position.append(slice(prev_word_finish, prev_word_finish + len(clear_inputs)))
            prev_word_finish += len(clear_inputs)
        cat_tensors = torch.cat([torch.tensor([2], dtype=torch.int32)] + inputs_ids_words + [torch.tensor([3], dtype=torch.int32)])
        # 2 - '[CLS]', 3 - '[SEP]'
        cat_tensors = cat_tensors.reshape(len(cat_tensors), 1)
        #print(self.tokenizer.convert_ids_to_tokens(cat_tensors))
        with torch.no_grad():
            outputs = self.model(**{'input_ids':cat_tensors.to(self.model.device)})
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
        embeddings = list(embeddings)[1:-1]
        word_embeds = []
        for word_slice in word_start_position:
            word_embeds.append(embeddings[word_slice])
        word_first_embeds = [e[0] for e in word_embeds]
        assert len(sent) == len(word_first_embeds)
        for e in word_first_embeds:
            assert e.shape == torch.Size([312])
        return word_first_embeds
