

paths = ["/home/h-asano/sasaki_work/data/ner/test.txt","/home/h-asano/sasaki_work/data/ner/train.txt","/home/h-asano/sasaki_work/data/ner/valid.txt"]
words = set()
for path in paths:
    for i, line in enumerate(open(path)):
        if i == 0:
            continue
        col = line.strip.split()

        if len(col) != 0:
            words.add(col[0])

# from IPython.core.debugger import Pdb; Pdb().set_trace()

for word in words:
    print(word)


import argparse
import json
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str)
args = parser.parse_args()

word_set = set()
for line in open(args.file):
    data = json.loads(line)
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']

    words1 = word_tokenize(sentence1)
    words2 = word_tokenize(sentence2)
    from IPython.core.debugger import Pdb; Pdb().set_trace()
    # words1 = sentence1.strip('').split()
    # words2 = sentence2.strip().split()

    for word in words1:
        word_set.add(word.lower())

    for word in words2:
        word_set.add(word.lower())


for word in word_set:
    print(word)

    # from IPython.core.debugger import Pdb; Pdb().set_trace()


import sys
for line in sys.stdin:
    if len(line.strip.split()) != 301:
        print(line)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vec_path', type=str)
parser.add_argument('--dim', type=int, default=300)
args = parser.parse_args()
import codecs
import numpy as np
with codecs.open(args.vec_path, "r", 'utf-8', errors='replace') as input_data:
    for i, line in enumerate(input_data):
        if i == 0:
            print(line.strip())
        else:
            from IPython.core.debugger import Pdb; Pdb().set_trace()
            col = line.strip('\n').rsplit(' ', args.dim)
            word = col[0]
            vec = np.array(col[1:])
            norm = np.linalg.norm(vec)
            if norm != 0:
                vec /= norm
            else:
                pass



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--org_config', type=str)
parser.add_argument('--embed_path', type=str) 
parser.add_argument('--dim', type=str, default=300)
parser.add_argument('--epoch', type=str, default=75)
parser.add_argument('--cuda_device', type=str, default=0)

args = parser.parse_args()

for line in open(args.org_config):
    line = line.strip('\n').replace("EMBED_PATH", args.embed_path)
    line = line.replace("EPOCH", args.epoch)
    line = line.replace("CUDA_DEVICE", args.cuda_device)
    print(line)



import glob

print(glob.glob('./train.sh.e*'))





