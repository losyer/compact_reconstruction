# coding: utf-8
import json, sys, argparse, os, codecs
from datetime import datetime
from utils import get_total_line
from collections import defaultdict

def get_ngram(word, nmax, nmin):
    all_ngrams =[]
    word = '^' + word + '@'
    N = len(word)
    f =lambda x,n :[x[i:i+n] for i in range(len(x)-n+1)]
    for n in range(N):
        if n+1 < nmin or n+1 > nmax:
            continue
        ngram_list = f(word, n+1)
        all_ngrams += ngram_list
    return all_ngrams

def main(args):

    total_line = get_total_line(path=args.ref_vec_path, test=args.test)

    print('create ngram frequency dictionary ...', flush=True)
    idx_freq_dic = defaultdict(int)
    with codecs.open(args.ref_vec_path, "r", 'utf-8', errors='replace') as input_data:
        for i, line in enumerate(input_data):

            if i % int(total_line/10) == 0:
                print('{} % done'.format(round(i / (total_line/100))), flush=True)

            if i == 0:
                col = line.strip('\n').split()
                vocab_size, dim = int(col[0]), int(col[1])
            else:
                col = line.strip(' \n').rsplit(' ', dim)
                assert len(col) == dim+1

                word = col[0]
                # if ' ' in word:
                #     from IPython.core.debugger import Pdb; Pdb().set_trace()
                if len(word) > 30:
                    continue
                ngrams = get_ngram(word, args.n_max, args.n_min)

                for ngram in ngrams:
                    idx_freq_dic[ngram] += 1

            if args.test and i > 1000:
                break

    print('create ngram frequency dictionary ... done', flush=True)

    # save
    print('save ... ', flush=True)
    fo = open(args.output, 'w')
    for ngram, freq in idx_freq_dic.items():
        fo.write('{} {}\n'.format(ngram, freq))
    fo.close()
    print('save ... done', flush=True)
    

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.add_argument('--n_max', type=int, default=30, help='')
    parser.add_argument('--n_min', type=int, default=3, help='')

    # data path
    parser.add_argument('--ref_vec_path', type=str, default="")
    parser.add_argument('--output', type=str, default="")

    args = parser.parse_args()
    main(args)

