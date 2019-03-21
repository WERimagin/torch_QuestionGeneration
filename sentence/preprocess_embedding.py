##gloveを使った単語ベクトルの生成
##python2に対して処理を行った(map)
##glove.840.300dの処理を行った

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip
from six.moves import cPickle


import numpy as np
from tqdm import tqdm

TRANSLATE = {
    "-lsb-" : "[",
    "-rsb-" : "]",
    "-lrb-" : "(",
    "-rrb-" : ")",
    "-lcb-" : "{",
    "-rcb-" : "}",
    "-LSB-" : "[",
    "-RSB-" : "]",
    "-LRB-" : "(",
    "-RRB-" : ")",
    "-LCB-" : "{",
    "-RCB-" : "}",
}


def parse_args(description = "I am lazy"):
    import argparse
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("--embedding", type = str, default = "./data_qg/glove.6B.100d.txt", required = True)
    parser.add_argument("--dict", type = str, required = True)
    parser.add_argument("--output", type = str, required = True)
    parser.add_argument("--seed", type = int, default = 19941023)
    args = parser.parse_args()
    np.random.seed(args.seed)
    return args


def main():
    args = parse_args()

    word2embedding = {}
    dimension = None
    with open(args.embedding, "r") as input_file:
        for i,line in tqdm(enumerate(input_file)):
            line = line.split()
            #一番上を単語とすると、. . .みたいなスペース混じりを処理できない。
            #よって下から300番目までをベクトルの数値として扱う
            word=" ".join(line[:-300])
            vec=line[-300:]
            word2embedding[word] = np.asarray(list(map(float, vec)))
            #word2embedding[line[0]] = np.asarray(list(map(float, line[1 : ])))
            dimension = len(line) - 1


    with open(args.dict, "r") as input_file:
        words = [ line.split()[0] for line in input_file ]

    embedding = np.random.uniform(low = -1.0 / 3, high = 1.0 / 3, size = (len(words), dimension))
    embedding = np.asarray(embedding, dtype = np.float32)
    unknown_count = 0
    for i, word in tqdm(enumerate(words)):
        if word in TRANSLATE:
            word = TRANSLATE[word]
        done = False
        for w in (word, word.upper(), word.lower()):
            if w in word2embedding:
                embedding[i] = word2embedding[w]
                done = True
                break
        if not done:
            print("Unknown word: %s" % (word, ))
            unknown_count += 1

    np.save(args.output, embedding)
    print("Total unknown: %d" % (unknown_count, ))


if __name__ == "__main__":
    main()
