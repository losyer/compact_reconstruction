# coding: utf-8
import argparse
import sys
from tqdm import tqdm

def main(fi, args):
    ngram_dic = {}
    f =lambda x,n :[x[i:i+n] for i in range(len(x)-n+1)]# print(f("I am an NLPer",2))

    print('create ngram_dic')
    # for i, line in tqdm(enumerate(open(args.input)), total=26125164):
    for i, line in tqdm(enumerate(open(args.input)), total=28422404):
        col = line.strip().split()
        if len(col) != 2:
            continue
        freq, word = int(col[0]), col[1]
        if args.add_header:
            # word = '^' + word + '@'
            word = word + '@'
        N = args.N
        if len(word) < N:
            N = len(word)
        for n in range(N):
            ngram_list = f(word, n+1)
            for ngram in ngram_list:
                if ngram in ngram_dic:
                    ngram_dic[ngram] += freq
                else:
                    ngram_dic[ngram] = freq
        
        # if i == 1000:
        #     break
    print('done')

    print('sort and write')
    fo = open(args.output, 'w')
    for k, v in sorted(ngram_dic.items(), key=lambda x:x[1], reverse=True):
        fo.write('{} {}\n'.format(k,v))
    fo.close()
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_header', action="store_true")
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--N', default=10, type=int)
    args = parser.parse_args()
    fi = sys.stdin
    main(fi, args)
