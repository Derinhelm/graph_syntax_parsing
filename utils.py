from collections import Counter
import re
import time
import random


from project_logging import logging

class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None,
        deps=None, misc=None):

        self.id = id
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

        self.pred_pos = None
        self.pred_cpos = None


    def __str__(self):
        '''values = [str(self.id), self.form, self.lemma, \
                  self.pred_cpos if self.pred_cpos else self.cpos,\
                  self.pred_pos if self.pred_pos else self.pos,\
                  self.feats, str(self.pred_parent_id) if self.pred_parent_id \
                  is not None else str(self.parent_id), self.pred_relation if\
                  self.pred_relation is not None else self.relation, \
                  self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])'''
        return self.form + " " + str(self.id)

class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None # TODO: зачем?
            root.parent = None
            root.pred_parent_id = None
            root.pred_relation = None
            root.vecs = None

    def __len__(self):
        return len(self.roots)


    def Attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]

    def __str__(self):
        return " ".join(map(str, self.roots))


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) \
                  for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and \
                        unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and \
                        unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id]-=1
                forest.Attach(i, i+1)
                break

    return len(forest.roots) == 1


def get_irels(data):
    """
    Collect frequencies of words, cpos, pos and deprels + languages.
    """

    # could use sets directly rather than counters for most of these,
    # but having the counts might be useful in the future or possibly for debugging etc
    relCount = Counter()

    for sentence in data:
        for node in sentence:
            if isinstance(node, ConllEntry):
                relCount.update([node.relation])

    return list(relCount.keys())


def generate_root_token():
    return ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1,
        'rroot', '_', '_')


def read_conll(filename, drop_nproj=False, train=True):
    fh = open(filename,'r',encoding='utf-8')
    logging.info(f"Reading {filename}")
    ts = time.time()
    dropped = 0
    sents_read = 0
    sentences = []
    tokens = [generate_root_token()]
    words = set() # all words from the dataset file
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '': # empty line, add sentence to list or yield
            if len(tokens) > 1:
                sents_read += 1
                conll_tokens = [t for t in tokens if isinstance(t,ConllEntry)]
                if not drop_nproj or isProj(conll_tokens):
                    # keep going if it's projective or we're not dropping non-projective sents
                    if train:
                        inorder_tokens = inorder(conll_tokens)
                        for i,t in enumerate(inorder_tokens):
                            t.projective_order = i
                        for tok in conll_tokens:
                            tok.rdeps = [i.id for i in conll_tokens if i.parent_id == tok.id]
                            if tok.id != 0:
                                tok.parent_entry = [i for i in conll_tokens if i.id ==\
                                                     tok.parent_id][0]
                    sentences.append(tokens)
                else:
                    logging.debug('Non-projective sentence dropped')
                    dropped += 1
            tokens = [generate_root_token()]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]: 
            # a comment line, add to tokens as is
                tokens.append(line.strip())
            else: # an actual ConllEntry, add to tokens
                if tok[2] == "_":
                    tok[2] = tok[1].lower()
                lemma = tok[2]
                words.add(lemma)
                token = ConllEntry(int(tok[0]), tok[1], lemma, tok[4], tok[3], tok[5], \
                    int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])

                tokens.append(token)

# deal with case where there are still tokens, that aren`t in sentences list
# e.g. when there is no newline at end of file
    if len(tokens) > 1:
        sentences.append(tokens)

    logging.debug(f'{sents_read} sentences read')

    te = time.time()
    logging.info(f'Time: {te-ts:.2g}s')
    return sentences, words


def write_conll(fn, conll_gen):
    logging.info(f"Writing to {fn}")
    sents = 0
    with open(fn, 'w', encoding='utf-8') as fh:
        for sentence in conll_gen:
            sents += 1
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')
        logging.debug(f"Wrote {sents} sentences")


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def inorder(sentence):
    queue = [sentence[0]]
    def inorder_helper(sentence,i):
        results = []
        left_children = [entry for entry in sentence[:i] if entry.parent_id == i]
        for child in left_children:
            results += inorder_helper(sentence,child.id)
        results.append(sentence[i])

        right_children = [entry for entry in sentence[i:] if entry.parent_id == i ]
        for child in right_children:
            results += inorder_helper(sentence,child.id)
        return results
    return inorder_helper(sentence,queue[0].id)


def set_seeds():
    python_seed = 1
    logging.debug("Using default Python seed")
    random.seed(python_seed)


def generate_seed():
    return random.randint(0,10**9) 
# this range seems to work for Dynet and Python's random function
