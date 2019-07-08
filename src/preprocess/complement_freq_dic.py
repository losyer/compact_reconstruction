# coding:utf-8
# python complement_freq_dic.py --org_freq_path /home/h-asano/sasaki_work/fastText-0.1.0/freq_count.LDC_enwiki_clean.txt.dlDOC.minC10.txt  --target_path /home/h-asano/sasaki_work/processed.txt.new
import argparse
import codecs
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--target_path', type=str)
parser.add_argument('--org_freq_path', type=str)
args = parser.parse_args()

print("create word to frequency dictionary ...", file=sys.stderr, flush=True)
word_to_freq_dic = {}
index_to_word = {}
for i, line in enumerate(codecs.open(args.org_freq_path, "r", 'utf-8', errors='replace')):
    col = line.strip().split("\t")
    assert len(col) == 2
    word, freq = col[0], int(col[1])
    word_to_freq_dic[word] = (i, freq)
    index_to_word[i] = word
print("create word to frequency dictionary ... done",file=sys.stderr, flush=True)

pre_word = ','
for i, line in enumerate(codecs.open(args.target_path, "r", 'utf-8', errors='replace')):
    if i == 0:
        continue
    word = line.strip().split()[0]
    try:
        freq = word_to_freq_dic[word][1]
        pre_word = word
    except:
        pre_word_index = word_to_freq_dic[pre_word][0]
        pre_word_freq = word_to_freq_dic[pre_word][1]
        next_word = index_to_word[pre_word_index+1]
        next_word_freq = word_to_freq_dic[next_word][1]

        freq = int((pre_word_freq + next_word_freq)/2)
    

    print('{}\t{}'.format(word, freq))


