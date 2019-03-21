import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from collections import defaultdict


def head_find(src,tgt):
    q_head=["what","how","who","when","which","where","why","whose","is","are","was","were","do","did","does"]
    src_tokens=word_tokenize(src)
    tgt_tokens=word_tokenize(tgt)
    true_head="<none>"
    for h in q_head:
        if h in tgt_tokens:
            true_head=h
            break
    src_tokens.append("<interr>")
    src_tokens.append(true_head)
    src=" ".join(src_tokens)
    return src



#tgt_path="tgt_test.txt"
#pred_path="pred_test.txt"
#src_path="../data/processed/src-train.txt"
#tgt_path="../data/processed/tgt-train.txt"
#src_modify_path="../data/processed/src-train-modify.txt"
src_path="../data/processed/src-dev.txt"
tgt_path="../data/processed/tgt-dev.txt"
src_modify_path="../data/processed/src-dev-modify.txt"

src=[]
target=[]
src2=[]

with open(src_path)as f:
    for line in f:
        src.append(line[:-1])

with open(tgt_path)as f:
    for line in f:
        target.append(line[:-1])

for s,t in tqdm(zip(src,target)):
    s=head_find(s,t)
    src2.append(s)

with open(src_modify_path,"w")as f:
    for s in src2:
        f.write(s+"\n")
