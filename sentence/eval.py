import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from collections import defaultdict

#tgt_path="tgt_test.txt"
#pred_path="pred_test.txt"
src_path="../data/processed/src-dev.txt"
tgt_path="../data/processed/tgt-dev.txt"
pred_path="pred.txt"

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

target_dict=defaultdict(lambda: [])
predict_dict=defaultdict(str)

for s,t,p in zip(src,target,predict):
    target_dict[s].append(t)
    predict_dict[s]=p

print("size:{}\n".format(len(target)))

score_sum_bleu1=0
score_sum_bleu2=0
for s in src:
    t=target_dict[s]
    p=predict_dict[s]
    score = sentence_bleu(t,p,weights=(1,0,0,0))
    score_sum_bleu1+=score
    print(t,p,score)
    score = sentence_bleu(t,p,weights=(0,1,0,0))
    score_sum_bleu2+=score


print(score_sum_bleu1/len(target),len(target))
print(score_sum_bleu2/len(target),len(target))
