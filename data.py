from utils import set_seeds, read_conll
from run import run
from embeddings import create_embeds, load_embeds, TinyBertEmbeddingCreator

def get_real_data(embed_pickle_using, context_embed, prefix, train_a_using=True,
                  train_b_using=True, train_c_using=True):
    if train_a_using:
        train_a, train_words_a = read_conll(prefix + 'ru_syntagrus-ud-train-a.conllu')
    else:
        train_a, train_words_a = [], set()
    if train_b_using:
        train_b, train_words_b = read_conll(prefix + 'ru_syntagrus-ud-train-b.conllu')
    else:
        train_b, train_words_b = [], set()
    if train_c_using:
        train_c, train_words_c = read_conll(prefix + 'ru_syntagrus-ud-train-c.conllu')
    else:
        train_c, train_words_c = [], set()
    train = train_a + train_b + train_c
    val, val_words = read_conll(prefix + 'ru_syntagrus-ud-dev.conllu')
    test, test_words = read_conll(prefix + 'ru_syntagrus-ud-test.conllu')
    if context_embed:
        embeds = ("context", TinyBertEmbeddingCreator())
    else:
        if embed_pickle_using:
            embeds = load_embeds(prefix + 'embeds.pickle')
        else:
            all_words = train_words_a | train_words_b | train_words_c | val_words | test_words
            embeds = create_embeds(all_words)
        embeds = ("independent", embeds)
    return train, val, test, embeds

def get_short_data(embed_pickle_using, context_embed, prefix):
    train, train_words = read_conll(prefix + 'ru_syntagrus-ud-train.conllu')
    val, val_words = read_conll(prefix + 'ru_syntagrus-ud-dev.conllu')
    test, test_words = read_conll(prefix + 'ru_syntagrus-ud-test.conllu')
    if context_embed:
        embeds = ("context", TinyBertEmbeddingCreator())
    else:
        if embed_pickle_using:
            embeds = load_embeds(prefix + 'embeds.pickle')
        else:
            all_words = train_words | val_words | test_words
            embeds = create_embeds(all_words)
        embeds = ("independent", embeds)
    return train, val, test, embeds

def get_data(real_dataset=False, embed_pickle_using=True,
             context_embed=True,
             colab=False, prefix="", train_a_using=True,
                  train_b_using=True, train_c_using=True):
# colab=True - run on colab
    if colab:
        if real_dataset:
            return get_real_data(embed_pickle_using,
                                 context_embed,
                                 prefix + '/UD_Russian-SynTagRus/',
                                 train_a_using, train_b_using, train_c_using)
        else:
            return get_short_data(embed_pickle_using,
                                  context_embed,
                                  prefix + '/UD_Russian-SynTagRus-small/')
    else:
        if real_dataset:
            return get_real_data(embed_pickle_using,
                                 context_embed,
                                 prefix + 'UD_Russian-SynTagRus/',
                                 train_a_using, train_b_using, train_c_using)
        else:
            return get_short_data(embed_pickle_using,
                                  context_embed,
                                  prefix + 'UD_Russian-SynTagRus-small/')
