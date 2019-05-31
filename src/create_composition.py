# -*- coding: utf-8 -*-
import sys, json, os, argparse, codecs
import numpy as np

def hash( str , prime=0x01000193):
    hval = 0x811c9dc5
    # fnv_32_prime = 0x01000193
    fnv_32_prime = prime
    uint32_max = 2 ** 32
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * fnv_32_prime) % uint32_max
    return hval

def create_composition_bpe_with_hash(word, pos_to_subwordlist, pos, index_only=False, filter_flag=False):
    subword_idx =[]
    subword_list = pos_to_subwordlist[pos]
    for subword in subword_list:
        subword = 'BPE-' + subword
        hashed_idx = hash(subword)
        if index_only:
            subword_idx.append(hashed_idx)
        else:
            subword_idx.append((hashed_idx, subword))
    return subword_idx

def create_composition_ngram_with_hash(word, nmax, nmin, index_only=False, bucket_size=1000000, filter_flag=False, multi_hash='', ngram_dic=None):
    ngram_idx =[]
    word = '^' + word + '@'
    N = len(word)
    f =lambda x,n :[x[i:i+n] for i in range(len(x)-n+1)]
    for n in range(N):
        if n+1 == N and filter_flag:
            continue
        if n+1 < nmin or n+1 > nmax:
            continue
        ngram_list = f(word, n+1)
        for ngram in ngram_list:
            if ngram_dic != None:
                if ngram in ngram_dic:
                    pass
                else:
                    continue    

            hashed_idx = hash(ngram)
            if index_only:
                ngram_idx.append(hashed_idx)
            else:
                ngram_idx.append((hashed_idx, ngram))
            if multi_hash == 'two' or multi_hash == 'three' or multi_hash == 'four':
                ngram_a = ngram + '-two'
                hashed_idx_a = hash(ngram_a)
                if index_only:
                    ngram_idx.append(hashed_idx_a)
                else:
                    ngram_idx.append((hashed_idx_a, ngram_a))

            if multi_hash == 'three' or multi_hash == 'four':
                ngram_b = ngram + '-three'
                hashed_idx_b = hash(ngram_b)
                if index_only:
                    ngram_idx.append(hashed_idx_b)
                else:
                    ngram_idx.append((hashed_idx_b, ngram_b))

            if multi_hash == 'four':
                ngram_c = ngram + '-four'
                hashed_idx_c = hash(ngram_c)
                if index_only:
                    ngram_idx.append(hashed_idx_c)
                else:
                    ngram_idx.append((hashed_idx_c, ngram_c))

    return ngram_idx

def create_composition_skip_ngram(word, ngram_dic, index_only=False, limit_size=0, ngram_dic_pos=0):
    try:
        ngram_idx = ngram_dic[ngram_dic_pos]
    except:
        try:
            print('WARNING: word = {}, ngram_dic_pos = {}'.format(word, ngram_dic_pos))
        except:
            # print('WARNING: ngram_dic_pos = {}'.format(ngram_dic_pos))
            pass
        return []
    return ngram_idx

def create_composition_ngram(word, ngram_dic, nmax, nmin, limit_size, index_only=False, filter_flag=False):
    ngram_idx =[]
    word = '^' + word + '@'
    N = len(word)
    f =lambda x,n :[x[i:i+n] for i in range(len(x)-n+1)]
    for n in range(N):
        if n+1 == N and filter_flag:
            continue
        if n+1 < nmin or n+1 > nmax:
            continue
        ngram_list = f(word, n+1)
        for ngram in ngram_list:
            try:
                idx = ngram_dic[ngram]
            except:
                continue
            if idx < limit_size:
                if index_only:
                    ngram_idx.append(idx)
                else:
                    ngram_idx.append((idx, ngram))
            else:
                continue
    return ngram_idx

def load_pos_to_subwordlist(file_path, test=False):
    print('load pos_to_subwordlist file ...', flush=True)
    pos_to_subwordlist = {}
    for i, line in enumerate(codecs.open(file_path, "r", 'utf-8', errors='replace')):
        subword_list = line.strip().split('\t')
        pos_to_subwordlist[i] = subword_list
        if test and i > 10000:
            break
    print('load pos_to_subwordlist file ... ... done', flush=True)
    return pos_to_subwordlist

def load_codecs_skip_ngram(codecs_path, limit_size=1000000, unique=True, codecs_type=0, test=False):
    print('load codecs file ...', flush=True)
    ngram_dic = {}
    error_count = 0
    for i, line in enumerate(open(codecs_path)):
        col   = line.strip().split('\t')
        num   = int(col[0])
        flist1 = [int(i) for i in col[1].split()]
        flist = [i for i in flist1 if i < limit_size ]        
        ngram_dic[num] = flist
        if len(flist) == 0:
            error_count += 1
            sys.stderr.write('#ERROR {} | {} {} | {} {} \n'.format(num, len(flist), flist, len(flist1), flist1))
        else:
            ##sys.stderr.write('{} | {} {} | {} {} \n'.format(num, len(flist), flist, len(flist1), flist1))
            pass
        if test and i > 10000:
            break

    print('#error count = {}'.format(error_count), flush=True)
    print('load codecs file ... done', flush=True)
    return ngram_dic

def load_codecs(codecs_path, limit_size, unique=True, codecs_type=0):
    print('load codecs file ...', flush=True)
    ngram_dic = {}
    # for i, line in enumerate(open(codecs_path)):
    for i, line in enumerate(codecs.open(codecs_path, "r", 'utf-8', errors='replace')):
        col = line.strip().split()
        ngram = col[0]
        if unique and ngram.startswith('^') and ngram.endswith('@'):
            continue
        if codecs_type == 1 and len(ngram) != 1:
            continue
        if codecs_type == 2 and len(ngram) > 2:
            continue
        # ngram_dic[ngram] = i
        ngram_dic[ngram] = len(ngram_dic)
        if len(ngram_dic) == limit_size:
            break
    print('load codecs file ... done', flush=True)
    print('###len(ngram_dic) = {}'.format(len(ngram_dic)), flush=True)
    return ngram_dic

def main(args):
    # ngram_dic = load_codecs(args.codecs_path)
    # ngram_dic = load_codecs(args.codecs_path, unique=True)
    word = 'ultimate'
    pos = 100
    print('create_composition')
    # # ngram_idx = create_composition_ngram(word, ngram_dic)
    subword_idx = create_composition_ngram_with_hash(word, 30, 1)
    # print(ngram_idx)
    # print('done')

    # pos_to_subwordlist = load_pos_to_subwordlist(args.codecs_path)
    # ssubword_idx = create_composition_bpe_with_hash(word, pos_to_subwordlist, pos)
    print(subword_idx)

    # create_composition_ngram('ultimate', ngram_dic,index_only=False, limit_size=999999)
    # from IPython.core.debugger import Pdb; Pdb  ().set_trace()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--codecs_path', type=str)
    args = parser.parse_args()
    main(args)


