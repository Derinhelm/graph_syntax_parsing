from uuparser.utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
import time, random
import numpy as np
from copy import deepcopy
from collections import defaultdict
import json
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim

import tqdm

from uuparser import utils

class ArcHybridLSTM:
    def __init__(self, vocab, options):

        # import here so we don't load Dynet if just running parser.py --help for example
        from uuparser.multilayer_perceptron import MLP

        global LEFT_ARC, RIGHT_ARC, SHIFT, SWAP
        LEFT_ARC, RIGHT_ARC, SHIFT, SWAP = 0,1,2,3

        self.irels, treebanks = vocab

        self.activation = options.activation

        self.mlp_in_dims = 30 # TODO: Create a logical value.

        self.unlabeled_MLP = MLP(self.mlp_in_dims, options.mlp_hidden_dims,
                                 options.mlp_hidden2_dims, 4, self.activation)
        self.labeled_MLP = MLP(self.mlp_in_dims, options.mlp_hidden_dims,
                               options.mlp_hidden2_dims,2*len(self.irels)+2, self.activation)


        self.unlabeled_optimizer = optim.Adam(self.unlabeled_MLP.parameters(), lr=options.learning_rate)
        self.labeled_optimizer = optim.Adam(self.labeled_MLP.parameters(), lr=options.learning_rate)

        self.oracle = options.oracle

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.k

    def create_configuration_representation(self, stack, buf): 
        # TODO: add sentence for representation creating
        #topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [] for i in range(self.k) ]
        #topBuffer = [ buf.roots[i].lstms if len(buf) > i else [] for i in range(1) ]

        #input = dy.concatenate(list(chain(*(topStack + topBuffer))))

        return torch.randn(self.mlp_in_dims)

    def __evaluate(self, stack, buf, train):
        """
        ret = [left arc,
               right arc
               shift]

        RET[i] = (rel, transition, score1, score2) for shift, l_arc and r_arc
         shift = 2 (==> rel=None) ; l_arc = 0; r_acr = 1

        ret[i][j][2] ~= ret[i][j][3] except the latter is a dynet
        expression used in the loss, the first is used in rest of training
        """

        input = self.create_configuration_representation(stack, buf)
        output = self.unlabeled_MLP(input)
        routput = self.labeled_MLP(input)


        #scores, unlabeled scores
        scrs, uscrs = routput, output

        #transition conditions
        left_arc_conditions = len(stack) > 0
        right_arc_conditions = len(stack) > 1
        shift_conditions = buf.roots[0].id != 0
        swap_conditions = len(stack) > 0 and stack.roots[-1].id < buf.roots[0].id

        if not train:
            #(avoiding the multiple roots problem: disallow left-arc from root
            #if stack has more than one element
            left_arc_conditions = left_arc_conditions and not (buf.roots[0].id == 0 and len(stack) > 1)

        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        uscrs2 = uscrs[2]
        uscrs3 = uscrs[3]

        if train:
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            output3 = output[3]


            ret = [ [ (rel, LEFT_ARC, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2 ] + output2) for j, rel in enumerate(self.irels) ] if left_arc_conditions else [],
                   [ (rel, RIGHT_ARC, scrs[3 + j * 2] + uscrs3, routput[3 + j * 2 ] + output3) for j, rel in enumerate(self.irels) ] if right_arc_conditions else [],
                   [ (None, SHIFT, scrs[0] + uscrs0, routput[0] + output0) ] if shift_conditions else [] ,
                    [ (None, SWAP, scrs[1] + uscrs1, routput[1] + output1) ] if swap_conditions else [] ]
        else:
            s1,r1 = max(zip(scrs[2::2],self.irels))
            s2,r2 = max(zip(scrs[3::2],self.irels))
            s1 += uscrs2
            s2 += uscrs3
            ret = [ [ (r1, LEFT_ARC, s1) ] if left_arc_conditions else [],
                   [ (r2, RIGHT_ARC, s2) ] if right_arc_conditions else [],
                   [ (None, SHIFT, scrs[0] + uscrs0) ] if shift_conditions else [] ,
                    [ (None, SWAP, scrs[1] + uscrs1) ] if swap_conditions else [] ]
        return ret


    def Save(self, filename):
        unlab_filename = filename + 'unlab'
        lab_filename = filename + 'lab'
        logger.info(f'Saving unlabeled model to {unlab_filename}')
        torch.save(unlabeled_MLP.state_dict(), PATH)
        logger.info(f'Saving labeled model to {lab_filename}')
        torch.save(labeled_MLP.state_dict(), PATH)
        self.model.save(filename)

    def Load(self, filename):
        unlab_filename = filename + 'unlab'
        lab_filename = filename + 'lab'
        logger.info(f'Loading unlabeled model from {unlab_filename}')

        logger.info(f'Loading labeled model from {lab_filename}')


        self.model.populate(filename)


    def apply_transition(self,best,stack,buf,hoffset):
        if best[1] == SHIFT:
            stack.roots.append(buf.roots[0])
            del buf.roots[0]

        elif best[1] == SWAP:
            child = stack.roots.pop()
            buf.roots.insert(1,child)

        elif best[1] == LEFT_ARC:
            child = stack.roots.pop()
            parent = buf.roots[0]

        elif best[1] == RIGHT_ARC:
            child = stack.roots.pop()
            parent = stack.roots[-1]

        if best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            #attach
            child.pred_parent_id = parent.id
            child.pred_relation = best[0]

    def calculate_cost(self,scores,s0,s1,b,beta,stack_ids):
        if len(scores[LEFT_ARC]) == 0:
            left_cost = 1
        else:
            left_cost = len(s0[0].rdeps) + int(s0[0].parent_id != b[0].id and s0[0].id in s0[0].parent_entry.rdeps)


        if len(scores[RIGHT_ARC]) == 0:
            right_cost = 1
        else:
            right_cost = len(s0[0].rdeps) + int(s0[0].parent_id != s1[0].id and s0[0].id in s0[0].parent_entry.rdeps)


        if len(scores[SHIFT]) == 0:
            shift_cost = 1
            shift_case = 0
        elif len([item for item in beta if item.projective_order < b[0].projective_order and item.id > b[0].id ])> 0:
            shift_cost = 0
            shift_case = 1
        else:
            shift_cost = len([d for d in b[0].rdeps if d in stack_ids]) + int(len(s0)>0 and b[0].parent_id in stack_ids[:-1] and b[0].id in b[0].parent_entry.rdeps)
            shift_case = 2


        if len(scores[SWAP]) == 0 :
            swap_cost = 1
        elif s0[0].projective_order > b[0].projective_order:
            swap_cost = 0
            #disable all the others
            left_cost = right_cost = shift_cost = 1
        else:
            swap_cost = 1

        costs = (left_cost, right_cost, shift_cost, swap_cost,1)
        return costs,shift_case


    def oracle_updates(self,best,b,s0,stack_ids,shift_case):
        if best[1] == SHIFT:
            if shift_case ==2:
                if b[0].parent_entry.id in stack_ids[:-1] and b[0].id in b[0].parent_entry.rdeps:
                    b[0].parent_entry.rdeps.remove(b[0].id)
                blocked_deps = [d for d in b[0].rdeps if d in stack_ids]
                for d in blocked_deps:
                    b[0].rdeps.remove(d)

        elif best[1] == LEFT_ARC or best[1] == RIGHT_ARC:
            s0[0].rdeps = []
            if s0[0].id in s0[0].parent_entry.rdeps:
                s0[0].parent_entry.rdeps.remove(s0[0].id)

    def Predict(self, treebanks, datasplit, options):
        reached_max_swap = 0

        data = utils.read_conll_dir(treebanks,datasplit)

        pbar = tqdm.tqdm(
            data,
            desc="Parsing",
            unit="sentences",
            mininterval=1.0,
            leave=False,
            disable=options.quiet,
        )

        for iSentence, osentence in enumerate(pbar,1):
            sentence = deepcopy(osentence)
            reached_swap_for_i_sentence = False
            max_swap = 2*len(sentence)
            iSwap = 0
            #self.feature_extractor.Init(options)
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
            for entry in conll_sentence:
                entry.vec = [] # There should be a vector from dynet here
            conll_sentence = [entry for entry in conll_sentence]
            stack = ParseForest([])
            buf = ParseForest(conll_sentence)

            hoffset = 1 if self.headFlag else 0

            for root in conll_sentence:
                root.relation = root.relation if root.relation in self.irels else 'runk'


            while not (len(buf) == 1 and len(stack) == 0):
                scores = self.__evaluate(stack, buf, False)
                best = max(chain(*(scores if iSwap < max_swap else scores[:3] )), key = itemgetter(2) )
                if iSwap == max_swap and not reached_swap_for_i_sentence:
                    reached_max_swap += 1
                    reached_swap_for_i_sentence = True
                    logger.debug(f"reached max swap in {reached_max_swap:d} out of {iSentence:d} sentences")
                self.apply_transition(best,stack,buf,hoffset)
                if best[1] == SWAP:
                    iSwap += 1

            #keep in memory the information we need, not all the vectors
            oconll_sentence = [entry for entry in osentence if isinstance(entry, utils.ConllEntry)]
            oconll_sentence = oconll_sentence[1:] + [oconll_sentence[0]]
            for tok_o, tok in zip(oconll_sentence, conll_sentence):
                tok_o.pred_relation = tok.pred_relation
                tok_o.pred_parent_id = tok.pred_parent_id
            yield osentence


    def Train(self, trainData, options):
        mloss = 0.0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ninf = -float('inf')


        beg = time.time()
        start = time.time()

        random.shuffle(trainData) # in certain cases the data will already have been shuffled after being read from file or while creating dev data
        logger.info(f"Length of training data: {len(trainData)}")

        errs = []

        #self.feature_extractor.Init(options)

        pbar = tqdm.tqdm(
            trainData,
            desc="Training",
            unit="sentences",
            mininterval=1.0,
            leave=False,
            disable=options.quiet,
        )

        for iSentence, sentence in enumerate(pbar,1):
            if iSentence % 100 == 0:
                loss_message = (
                    f'Processing sentence number: {iSentence}'
                    f' Loss: {eloss / etotal:.3f}'
                    f' Errors: {eerrors / etotal:.3f}'
                    f' Labeled Errors: {lerrors / etotal:.3f}'
                    f' Time: {time.time()-start:.3f}s'
                )
                logger.debug(loss_message)
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0
                lerrors = 0

            sentence = deepcopy(sentence) # ensures we are working with a clean copy of sentence and allows memory to be recycled each time round the loop

            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
            #self.feature_extractor.getWordEmbeddings(conll_sentence, True, options)
            for entry in conll_sentence:
                entry.vec = [] # There should be a vector from dynet here            
            stack = ParseForest([])
            buf = ParseForest(conll_sentence)
            hoffset = 1 if self.headFlag else 0

            for root in conll_sentence:
                root.relation = root.relation if root.relation in self.irels else 'runk'

            while not (len(buf) == 1 and len(stack) == 0):
                scores = self.__evaluate(stack, buf, True)

                #to ensure that we have at least one wrong operation
                scores.append([(None, 4, ninf ,None)])

                stack_ids = [sitem.id for sitem in stack.roots]

                s1 = [stack.roots[-2]] if len(stack) > 1 else []
                s0 = [stack.roots[-1]] if len(stack) > 0 else []
                b = [buf.roots[0]] if len(buf) > 0 else []
                beta = buf.roots[1:] if len(buf) > 1 else []

                costs, shift_case = self.calculate_cost(scores,s0,s1,b,beta,stack_ids)

                bestValid = list(( s for s in chain(*scores) if costs[s[1]] == 0 and ( s[1] == SHIFT or s[1] == SWAP or  s[0] == s0[0].relation ) ))

                bestValid = max(bestValid, key=itemgetter(2))
                bestWrong = max(( s for s in chain(*scores) if costs[s[1]] != 0 or ( s[1] != SHIFT and s[1] != SWAP and s[0] != s0[0].relation ) ), key=itemgetter(2))

                #force swap
                if costs[SWAP]== 0:
                    best = bestValid
                else:
                    #select a transition to follow
                    # + aggresive exploration
                    #1: might want to experiment with that parameter
                    if bestWrong[1] == SWAP:
                        best = bestValid
                    else:
                        best = bestValid if ( (not self.oracle) or (bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong

                if best[1] == LEFT_ARC or best[1] ==RIGHT_ARC:
                    child = s0[0]

                #updates for the dynamic oracle
                if self.oracle:
                    self.oracle_updates(best,b,s0,stack_ids,shift_case)

                self.apply_transition(best,stack,buf,hoffset)

                if bestValid[2] < bestWrong[2] + 1.0:
                    loss = bestWrong[3] - bestValid[3]
                    mloss += 1.0 + bestWrong[2] - bestValid[2]
                    eloss += 1.0 + bestWrong[2] - bestValid[2]
                    errs.append(loss)

                #labeled errors
                if best[1] == LEFT_ARC or best[1] ==RIGHT_ARC:
                    if (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                        lerrors += 1
                        #attachment error
                        if child.pred_parent_id != child.parent_id:
                            eerrors += 1

                #??? when did this happen and why?
                if best[1] == 0 or best[1] == 2:
                    etotal += 1

            #footnote 8 in Eli's original paper
            if len(errs) > 50: # or True:
                self.labeled_optimizer.zero_grad()
                self.unlabeled_optimizer.zero_grad()
                eerrs = torch.sum(torch.tensor(errs, requires_grad=True))
                eerrs.backward()
                self.labeled_optimizer.step() # TODO Какой из оптимизаторов ???
                self.unlabeled_optimizer.step()
                errs = []
                lerrs = []

                #self.feature_extractor.Init(options)

        if len(errs) > 0:
            self.labeled_optimizer.zero_grad()
            self.unlabeled_optimizer.zero_grad()
            eerrs = torch.sum(torch.tensor(errs, requires_grad=True))
            eerrs.backward()
            self.labeled_optimizer.step() # TODO Какой из оптимизаторов ???
            self.unlabeled_optimizer.step()
            errs = []
            lerrs = []


        logger.info(f"Loss: {mloss/iSentence}")
        logger.info(f"Total Training Time: {time.time()-beg:.2g}s")
