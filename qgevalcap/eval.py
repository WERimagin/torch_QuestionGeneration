#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'xinya'

from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from collections import defaultdict
from argparse import ArgumentParser

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
            #(Meteor(),"METEOR"),
            #(Rouge(), "ROUGE_L"),
            # (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            print(score)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print "%s: %0.5f"%(m, sc)
                    output.append(sc)
            else:
                print "%s: %0.5f"%(method, score)
                output.append(score)
        return output

def eval(out_file, src_file, tgt_file, isDIn = False, num_pairs = 500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """
    #pairs:リスト、サイズはlen(sentence)、中身はpair
    #pair:sentence,question_target,question_predictが辞書の形で格納されている。


    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)


    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]


    ## eval
    from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    #res:key:sentence,value:prediction
    #gts:key:sentence,value:question
    #ただし、gtsの方は同じsentenceについてはquestionを一つのsentenceに与える
    #また、一つの文につき一つのpredictしか評価していない。->10000文の内4000文は評価していない。
    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for i,pair in enumerate(pairs[:]):
        key = pair['tokenized_sentence']
        res[key] = [pair['prediction'].encode('utf-8')]
        gts[key].append(pair['tokenized_question'].encode('utf-8'))

    print(len(pairs))
    print(len(res.items()))

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="../data/pred_modify.txt", help="output file to compare")
    parser.add_argument("-src", "--src_file", dest="src_file", default="../data/processed/src-dev.txt", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="../data/processed/tgt-dev.txt", help="target file")
    args = parser.parse_args()

    print "scores: \n"
    eval(args.out_file, args.src_file, args.tgt_file)
