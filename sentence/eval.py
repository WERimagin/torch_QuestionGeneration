import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from collections import defaultdict
import collections
import math

def compute_score(t,p):
    t_dict=collections.Counter(t)
    p_dict=collections.Counter(p)
    common=sum((t_dict & p_dict).values())
    return common

def compute_score_refs(ts,p):
    sum_dict=collections.Counter()
    p_dict=collections.Counter(p)
    for t in ts:
        t_dict=collections.Counter(t)
        sum_dict=(t_dict & p_dict) | sum_dict
    common=sum(sum_dict.values())
    return common

def ngram(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def head_find(tokens):
    q_head=["what","how","who","when","which","where","why","whose","is","are","was","were","do","did","does"]
    for h in q_head:
        if h in tokens:
            return h
    return "<none>"

def head_compare(tgt,pred):
    t=head_find(tgt)
    p=head_find(pred)
    return t==p

modify=False

#tgt_path="tgt_test.txt"
#pred_path="pred_test.txt"
"""
if modify==False:
    src_path="../data/processed/src-dev.txt"
    tgt_path="../data/processed/tgt-dev.txt"
    pred_path="../data/pred.txt"
if modify==True:
    src_path="../data/processed/src-dev-modify.txt"
    tgt_path="../data/processed/tgt-dev.txt"
    pred_path="../data/pred_modify.txt"

src_path="../data/open_nmt/squad-src-train.txt"
tgt_path="../data/open_nmt/squad-tgt-train.txt"
pred_path="../data/open_nmt/pred_train.txt"
"""
src_path="../data/open_nmt/squad-src-dev.txt"
tgt_path="../data/open_nmt/squad-tgt-dev.txt"
pred_path="../data/pred_test.txt"


src=[]
target=[]
predict=[]

with open(src_path)as f:
    for line in f:
        src.append(line[:-1])

with open(tgt_path)as f:
    for line in f:
        target.append(line[:-1])

with open(pred_path)as f:
    for line in f:
        predict.append(line[:-1])
"""
for i in range(len(target)):
    if i%500==0:
        s=src[i]
        t=target[i]
        p=predict[i]
        print(s)
        print(t)
        print(p)
        print()
"""
target=[word_tokenize(s) for s in target]
predict=[word_tokenize(s) for s in predict]
"""
count=0
for t,p in zip(target,predict):
    count+=head_compare(t,p)

print(count/len(target))

target_dict=collections.Counter()
predict_dict=collections.Counter()

for s in target:
    target_dict[head_find(s)]+=1
print(target_dict.most_common())

for s in predict:
    predict_dict[head_find(s)]+=1
print(predict_dict.most_common())
"""


#一文ずつ評価,corpusのサイズ考慮
if True:
    for n in range(1,2):
        score_sum=0
        count_target=0
        count_predict=0
        for i in tqdm(range(len(predict))):
            t=target[i]
            p=predict[i]
            if i<=100:
                print(" ".join(t))
                print(" ".join(p))
                print()
            #t=ngram(word_tokenize(t),n)
            #p=ngram(word_tokenize(p),n)
            score_sum+=compute_score(t,p)
            count_target+=len(t)
            count_predict+=len(p)
        penalty=math.exp(1-count_target/count_predict) if count_target>count_predict else 1
        score=penalty*score_sum/count_predict
        print(count_target,count_predict)
        print(score)


########################

#同じ文はまとめてtargetとして扱う。
#この手法は同じpredictについてもそれぞれ計算。元のはまとめて計算
#shortのoptionについてはほとんど一致
if True:
    print(len(src),len(target),len(predict))
    src=src[0:len(predict)]
    target=target[0:len(predict)]
    predict=predict[0:len(predict)]

    target_dict=defaultdict(lambda: [])
    predict_dict=defaultdict(str)
    src_set=set(src)

    for s,t,p in zip(src,target,predict):
        target_dict[s].append(t)
        predict_dict[s]=p

    print("size:{}\n".format(len(target)))

    score_sum=0
    count_target=0
    count_predict=0
    t_list=[]
    p_list=[]
    for i,s in tqdm(enumerate(src_set)):
        t=target_dict[s]
        p=predict_dict[s]
        score=compute_score_refs(t,p)
        score_sum+=score
        c_t=min(map(len,t))
        c_p=len(p)
        p_list.append(c_t)
        count_target+=c_t
        count_predict+=c_p
        #print(score,len(p))
        #print(p)


    print(count_target,count_predict)
    print(sum(sorted(p_list)[:-100:-1]))
    print(len(p_list))
    penalty=math.exp(1-count_target/count_predict) if count_target>count_predict else 1
    score=penalty*score_sum/count_predict
    print(score)


    print()
"""
######################################

#一文ずつ評価(sentence_bleu使用)
if False:
    score_sum_bleu1=0
    score_sum_bleu2=0
    for t,p in tqdm(zip(target,predict)):
        t=word_tokenize(t)
        p=word_tokenize(p)
        score = sentence_bleu([t],p,weights=(1,0,0,0))
        score_sum_bleu1+=score
        score = sentence_bleu([t],p,weights=(0,1,0,0))
        score_sum_bleu2+=score

    print(score_sum_bleu1/len(target),len(target))
    print(score_sum_bleu2/len(target),len(target))

"""
