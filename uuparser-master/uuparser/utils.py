from collections import defaultdict, Counter
import re
import os,time
from itertools import chain
from operator import itemgetter
import random
import json
import pathlib
import subprocess
import sys

from loguru import logger

import tqdm

UTILS_PATH = pathlib.Path(__file__).parent/"utils"

# a global variable so we don't have to keep loading from file repeatedly
iso_dict = {}
reverse_iso_dict = {}


class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None,
        deps=None, misc=None, treebank_id=None, proxy_tbank=None, language=None, char_rep=None):

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
        self.language = language

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


class Treebank(object):
    def __init__(self,trainfile,devfile,testfile):
        self.name = 'noname'
        self.trainfile = trainfile
        self.devfile = devfile
        self.dev_gold = devfile
        self.test_gold = testfile
        self.testfile = testfile
        self.outfilename = None
        self.proxy_tbank = None


class UDtreebank(Treebank):
    def __init__(self, treebank_info, options):
        """
        Read treebank info to a treebank object
        """
        self.name, self.iso_id = treebank_info
        if not options.shared_task:

            if options.datadir:
                data_path = os.path.join(options.datadir,self.name)
            else:
                data_path = os.path.join(options.testdir,self.name)

            if options.testdir:
                test_path = os.path.join(options.testdir,self.name)
            else:
                test_path = data_path

            if options.golddir:
                gold_path = os.path.join(options.golddir,self.name)
            else:
                gold_path = test_path

            self.trainfile = os.path.join(data_path, self.iso_id + "-ud-train.conllu")
            self.devfile = os.path.join(test_path, self.iso_id + "-ud-dev.conllu")
            self.testfile = os.path.join(test_path, self.iso_id + "-ud-test.conllu")
            self.test_gold = os.path.join(gold_path, self.iso_id + "-ud-test.conllu")
            self.dev_gold = os.path.join(gold_path, self.iso_id + "-ud-dev.conllu")
        else:
            if not options.predict:
                self.trainfile = os.path.join(options.datadir, self.iso_id + ".conllu")
            #self.testfile = os.path.join(options.testdir, self.iso_id + "-udpipe.conllu")
            #self.testfile = os.path.join(options.testdir, self.iso_id + ".conllu")
            self.testfile = os.path.join(options.testdir, self.iso_id + ".txt")
            if options.golddir:
                self.test_gold = os.path.join(options.golddir, self.iso_id + ".conllu")
            else:
                self.test_gold = self.testfile
            self.devfile = self.testfile
            self.dev_gold = self.test_gold
        self.outfilename = self.iso_id + '.conllu'


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
            root.lstms = None

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


def get_vocab(treebanks,datasplit,char_map={}):
    """
    Collect frequencies of words, cpos, pos and deprels + languages.
    """

    data = read_conll_dir(treebanks,datasplit,char_map=char_map)

    # could use sets directly rather than counters for most of these,
    # but having the counts might be useful in the future or possibly for debugging etc
    relCount = Counter()
    tbankCount = Counter() # note that one language can have several treebanks
    langCount = Counter()

    for sentence in data:
        for node in sentence:
            if isinstance(node, ConllEntry):
                relCount.update([node.relation])
                treebank_id = node.treebank_id
                tbankCount.update([treebank_id])

    return (list(relCount.keys()), list(tbankCount.keys()))


def load_iso_dict(json_file=UTILS_PATH/'ud_iso.json'):
    logger.debug(f"Loading ISO dict from {json_file}")
    global iso_dict
    ud_iso_file = open(json_file,encoding='utf-8')
    json_str = ud_iso_file.read()
    iso_dict = json.loads(json_str)


def load_reverse_iso_dict(json_file=UTILS_PATH/'ud_iso.json'):
    global reverse_iso_dict
    if not iso_dict:
        load_iso_dict(json_file=json_file)
    reverse_iso_dict = {v: k for k, v in iso_dict.items()}


# convert treebank to language by removing everything after underscore
def get_lang_from_tbank_name(tbank_name):

    # weird exceptions of separate langs with dashes
    m = re.match('UD_(Norwegian-(Bokmaal|Nynorsk))',tbank_name)
    if m:
        lang = m.group(1)
    else:
        m = re.search('^UD_(.*?)(-|$)',tbank_name)
        lang = m.group(1) if m else tbank_name

    return lang


def get_lang_from_tbank_id(tbank_id):
    if not tbank_id:
        return None

    if not reverse_iso_dict:
        load_reverse_iso_dict()
    return get_lang_from_tbank_name(reverse_iso_dict[tbank_id])


# from a list of treebanks, return those that match a particular language
def get_treebanks_from_lang(treebank_ids,lang):
   return [treebank_id for treebank_id in treebank_ids if get_lang_from_tbank_id(treebank_id) == lang]


def get_all_treebanks(options):

    if not iso_dict:
        load_iso_dict(options.json_isos)
    treebank_metadata = iso_dict.items()

    json_treebanks = [UDtreebank(ele, options) for ele in treebank_metadata]

    return json_treebanks


def read_conll_dir(treebanks,filetype,maxSize=-1,char_map={}):
    logger.debug("Max size for each corpus: ", maxSize)
    if filetype == "train":
        return chain(*(read_conll(treebank.trainfile, treebank.iso_id, treebank.proxy_tbank, maxSize, train=True, char_map=char_map) for treebank in treebanks))
    elif filetype == "dev":
        return chain(*(read_conll(treebank.devfile, treebank.iso_id, treebank.proxy_tbank, train=False, char_map=char_map) for treebank in treebanks))
    elif filetype == "test":
        return chain(*(read_conll(treebank.testfile, treebank.iso_id, treebank.proxy_tbank, train=False, char_map=char_map) for treebank in treebanks))


def generate_root_token(treebank_id, proxy_tbank, language):
    return ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1,
        'rroot', '_', '_',treebank_id=treebank_id, proxy_tbank=proxy_tbank,
        language=language)


def read_conll(filename, treebank_id=None, proxy_tbank=None, maxSize=-1, hard_lim=False, vocab_prep=False, drop_nproj=False, train=True, char_map={}):
    # hard lim means capping the corpus size across the whole training procedure
    # soft lim means using a sample of the whole corpus at each epoch
    fh = open(filename,'r',encoding='utf-8')
    logger.info(f"Reading {filename}")
    if vocab_prep and not hard_lim:
        maxSize = -1 # when preparing the vocab with a soft limit we need to use the whole corpus
    ts = time.time()
    dropped = 0
    sents_read = 0
    if treebank_id:
        language = get_lang_from_tbank_id(treebank_id)
    else:
        language = None
    tokens = [generate_root_token(treebank_id, proxy_tbank, language)]
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
            tokens = [generate_root_token(treebank_id, proxy_tbank, language)]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]: # a comment line, add to tokens as is
                tokens.append(line.strip())
            else: # an actual ConllEntry, add to tokens
                char_rep = tok[1] # representation to use in character model
                if language and language in char_map:
                    for char in char_map[language]:
                        char_rep = re.sub(char,char_map[language][char],char_rep)
                if tok[2] == "_":
                    tok[2] = tok[1].lower()
                token = ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9],treebank_id=treebank_id,proxy_tbank=proxy_tbank,language=language,char_rep=char_rep)

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


def write_conll_multiling(conll_gen, treebanks):
    tbank_dict = {treebank.iso_id:treebank for treebank in treebanks}
    cur_tbank = conll_gen[0][0].treebank_id
    outfile = tbank_dict[cur_tbank].outfilename
    fh = open(outfile,'w',encoding='utf-8')
    logger.info(f"Writing to {outfile}")
    for sentence in conll_gen:
        if cur_tbank != sentence[0].treebank_id:
            fh.close()
            cur_tbank = sentence[0].treebank_id
            outfile = tbank_dict[cur_tbank].outfilename
            fh = open(outfile,'w',encoding='utf-8')
            logger.info(f"Writing to {outfile}")
        for entry in sentence[1:]:
            fh.write(str(entry) + '\n')
        fh.write('\n')


def parse_list_arg(l):
    """Return a list of line values if it's a file or a list of values if it
    is a string"""
    if os.path.isfile(l):
        f = open(l, 'r', encoding='utf-8')
        return [line.strip("\n").split()[0] for line in f]
    else:
        return [el for el in l.split(" ")]


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def evaluate(gold,test,conllu):
    scoresfile = test + '.txt'
    logger.info(f"Writing to {scoresfile}")
    with open(scoresfile, "w") as scoresfile_stream:
        if not conllu:
            #os.system('perl src/utils/eval.pl -g ' + gold + ' -s ' + test  + ' > ' + scoresfile + ' &')
            subprocess.run(["perl", str(UTILS_PATH/"eval.pl"), "-g", gold, "-s", test], stdout=scoresfile_stream, check=True)
        else:
            subprocess.run(["python", str(UTILS_PATH/"evaluation_script/conll17_ud_eval.py"), "-v", "-w", str(UTILS_PATH/"evaluation_script/weights.clas"), gold, test], stdout=scoresfile_stream, check=True)
    score = get_LAS_score(scoresfile,conllu)
    return score


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


def get_LAS_score(filename, conllu=True):
    score = None
    with open(filename,'r',encoding='utf-8') as fh:
        if conllu:
            for line in fh:
                if re.match(r'^LAS',line):
                    elements = line.split()
                    score = float(elements[6]) # should extract the F1 score
        else:
            las_line = [line for line in fh][0]
            score = float(las_line.split('=')[1].split()[0])

    return score


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
    if hasattr(options,'char_map_file'):
        stored_opt.char_map_file = options.char_map_file
    if hasattr(stored_opt,'lang_emb_size'):
        stored_opt.tbank_emb_size = stored_opt.lang_emb_size


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