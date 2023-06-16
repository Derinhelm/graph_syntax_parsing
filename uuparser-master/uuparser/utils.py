from collections import defaultdict, Counter
import re
import os,time
from operator import itemgetter
import random
import json
import pathlib
import subprocess
import sys

from loguru import logger

import tqdm


UTILS_PATH = pathlib.Path(__file__).parent/"utils"


class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None,
        deps=None, misc=None, treebank_id=None, proxy_tbank=None, char_rep=None):

        self.id = id
        self.form = form
        self.char_rep = char_rep if char_rep else form
        self.norm = normalize(self.char_rep)
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
        self.treebank_id = treebank_id
        self.proxy_tbank = proxy_tbank

        self.pred_pos = None
        self.pred_cpos = None


    def __str__(self):
        values = [str(self.id), self.form, self.lemma, \
                  self.pred_cpos if self.pred_cpos else self.cpos,\
                  self.pred_pos if self.pred_pos else self.pos,\
                  self.feats, str(self.pred_parent_id) if self.pred_parent_id \
                  is not None else str(self.parent_id), self.pred_relation if\
                  self.pred_relation is not None else self.relation, \
                  self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
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


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
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


def generate_root_token(treebank_id, proxy_tbank):
    return ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1,
        'rroot', '_', '_',treebank_id=treebank_id, proxy_tbank=proxy_tbank)


def read_conll(filename, treebank_id=None, proxy_tbank=None, maxSize=-1, hard_lim=False, vocab_prep=False, drop_nproj=False, train=True):
    # hard lim means capping the corpus size across the whole training procedure
    # soft lim means using a sample of the whole corpus at each epoch
    fh = open(filename,'r',encoding='utf-8')
    logger.info(f"Reading {filename}")
    if vocab_prep and not hard_lim:
        maxSize = -1 # when preparing the vocab with a soft limit we need to use the whole corpus
    ts = time.time()
    dropped = 0
    sents_read = 0
    tokens = [generate_root_token(treebank_id, proxy_tbank)]
    yield_count = 0
    if maxSize > 0 and not hard_lim:
        sents = []
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '': # empty line, add sentence to list or yield
            if len(tokens)>1:
                sents_read += 1
                conll_tokens = [t for t in tokens if isinstance(t,ConllEntry)]
                if not drop_nproj or isProj(conll_tokens): # keep going if it's projective or we're not dropping non-projective sents
                    if train:
                        inorder_tokens = inorder(conll_tokens)
                        for i,t in enumerate(inorder_tokens):
                            t.projective_order = i
                        for tok in conll_tokens:
                            tok.rdeps = [i.id for i in conll_tokens if i.parent_id == tok.id]
                            if tok.id != 0:
                                tok.parent_entry = [i for i in conll_tokens if i.id == tok.parent_id][0]
                    if maxSize > 0:
                        if not hard_lim:
                            sents.append(tokens)
                        else:
                            yield tokens
                            yield_count += 1
                            if yield_count == maxSize:
                                logger.info(f"Capping size of corpus at {yield_count} sentences")
                                break
                    else:
                        yield tokens
                else:
                    logger.debug('Non-projective sentence dropped')
                    dropped += 1
            tokens = [generate_root_token(treebank_id, proxy_tbank)]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]: # a comment line, add to tokens as is
                tokens.append(line.strip())
            else: # an actual ConllEntry, add to tokens
                char_rep = tok[1] # representation to use in character model
                if tok[2] == "_":
                    tok[2] = tok[1].lower()
                token = ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9],treebank_id=treebank_id,proxy_tbank=proxy_tbank,char_rep=char_rep)

                tokens.append(token)

    if hard_lim and yield_count < maxSize:
        logger.warning(f'Unable to yield {maxSize} sentences, only {yield_count} found')

# TODO: deal with case where there are still unyielded tokens
# e.g. when there is no newline at end of file
#    if len(tokens) > 1:
#        yield tokens

    logger.debug(f'{sents_read} sentences read')

    if maxSize > 0 and not hard_lim:
        if len(sents) > maxSize:
            sents = random.sample(sents,maxSize)
            logger.debug(f"Yielding {len(sents)} random sentences")
        for toks in sents:
            yield toks

    te = time.time()
    logger.info(f'Time: {te-ts:.2g}s')


def write_conll(fn, conll_gen):
    logger.info(f"Writing to {fn}")
    sents = 0
    with open(fn, 'w', encoding='utf-8') as fh:
        for sentence in conll_gen:
            sents += 1
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')
        logger.debug(f"Wrote {sents} sentences")


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


def set_seeds(options):
    python_seed = 1
    logger.debug("Using default Python seed")
    random.seed(python_seed)


def generate_seed():
    return random.randint(0,10**9) # this range seems to work for Dynet and Python's random function

# for the most part, we want to send stored options to the parser when in
# --predict mode, however we want to allow some of these to be updated
# based on the command line options specified by the user at predict time
def fix_stored_options(stored_opt,options):

    stored_opt.predict = True
    # force language embedding needs to be passed to parser!
    stored_opt.forced_tbank_emb = options.forced_tbank_emb
    stored_opt.ext_emb_dir = options.ext_emb_dir
    stored_opt.ext_word_emb_file = options.ext_word_emb_file
    stored_opt.ext_char_emb_file = options.ext_char_emb_file
    stored_opt.max_ext_emb = options.max_ext_emb
    stored_opt.shared_task = options.shared_task


class TqdmCompatibleStream:
    """Wrapper around a file-like object (usually stderr) that will call
    `tqdm.write` if a progressbar is active.
    """

    def __init__(self, file=sys.stderr):
        self.file = file

    def write(self, x):
        if getattr(tqdm.tqdm, "_instances", []):
            # Avoid print() second call (useless \n)
            x = x.rstrip()
            if len(x) > 0:
                tqdm.tqdm.write(x, file=self.file)
        else:
            self.file.write(x)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)