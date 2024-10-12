import torch

from constants import EMBED_SIZE

def create_start_embeds(sentence, embeds):
    # sentence = [ConllEntry_root, ConllEntry_1, ConllEntry_2, ...]
    sentence[0].start_embed = torch.zeros(EMBED_SIZE) # TODO: temporary solution for root element
    
    if embeds[0] == "context": # context embeds are used
        sent_words = [w.form for w in sentence[1:]]
        #print("sent_words:", sent_words)
        embed_creator = embeds[1]
        sent_embeds = embed_creator.create_first_bert_embeddings_sent(sent_words)
        for i in range(len(sent_words)):
            sentence[i + 1].start_embed = sent_embeds[i]
    elif embeds[0] == "independent":
        # embeds from dict are used (embeds are generated for lexeme without context)
        for i, word in enumerate(sentence[1:]):
            sentence[i + 1].start_embed = embeds[1][word.lemma]
    else:
        print("Wrong embedding type:", embeds[0])
        import sys
        sys.exit(-1)
