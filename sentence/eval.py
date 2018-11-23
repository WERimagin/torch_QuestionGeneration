import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

#tgt_path="tgt_test.txt"
#pred_path="pred_test.txt"
tgt_path="../data/processed/tgt-dev.txt"
pred_path="pred.txt"

target=[]
predict=[]

with open(tgt_path)as f:
    for line in f:
        target.append(line)

with open(pred_path)as f:
    for line in f:
        predict.append(line)

target=[word_tokenize(sent) for sent in target]
predict=[word_tokenize(sent) for sent in predict]

print("size:{}\n".format(len(target)))

score_sum_bleu1=0
score_sum_bleu2=0
for t,p in tqdm(zip(target,predict)):
    score = sentence_bleu([t],p,weights=(1,0,0,0))
    score_sum_bleu1+=score
    score = sentence_bleu([t],p,weights=(0,1,0,0))
    score_sum_bleu2+=score

print(score_sum_bleu1/len(target),len(target))
print(score_sum_blue2/len(target),len(target))
