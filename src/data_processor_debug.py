# -*- coding: utf-8 -*-
import sys, json, os, codecs
import numpy as np
from collections import defaultdict
from itertools import groupby   
from chainer import cuda
from create_composition import create_composition_ngram, load_codecs

class DataProcessor(object):

    def __init__(self, args):
        self.ref_vec_path = args.ref_vec_path
        self.test = args.test
        self.limit_size = args.limit_size
        self.args = args

        self.ngram_dic = load_codecs(args.codecs_path, limit_size=args.limit_size, unique=args.unique)
        self.set_n_vocab_subword()
        self.filtering_words = self.load_filtering_words(args.filtering_words_path)

    def set_n_vocab_subword(self):
        if self.limit_size !=0:
            self.n_vocab_subword = self.limit_size
        else:
            self.n_vocab_subword = len(self.merge_dic)
        print('n_vocab_subword = ', self.n_vocab_subword)

    def load_filtering_words(self, path):
        if path == '':
            return set()
        else:
            print('loading filtering words')
            words = set()
            for line in codecs.open(path, "r", 'utf-8', errors='replace'):
                word = line.strip()
                words.add(word)
            print('done')
            return words

    def prepare_dataset(self):
        print("loading dataset...")
        self.train_data = self.load_dataset()

        if self.test:
            print("\ntiny dataset for quick test...")
            print('length of dataset =', len(self.train_data))
        print("\ndone")

    def load_dataset(self):

        dataset = []
        maxlen=450

        if not(self.test):
            print('get # of lines', flush=True)
            with codecs.open(self.ref_vec_path, "r", 'utf-8', errors='replace') as input_data:
                total_line = len([0 for _ in  input_data])
            print('done', flush=True)
            print('# of lines = {}'.format(total_line), flush=True)
        else:
            total_line = 1

        sum_idx = 0
        word_count = 0
        idx_freq_dic = defaultdict(int)
        word_subidx_dic = {}

        with codecs.open(self.ref_vec_path, "r", 'utf-8', errors='replace') as input_data:
            for i, line in enumerate(input_data):
                
                # print progress
                if self.args.raiden:
                    if i % (total_line/10) == 0:
                        print('{} % done'.format(i / (total_line/100)), flush=True)
                else:
                    sys.stderr.write('\rloading: %d' % i)

                col = line.strip().split()
                if i == 0:
                    vocab_size, dim = int(col[0]), int(col[1])
                else:
                    if len(col) != dim+1:
                        continue
                    # word = '^' + col[0] + '</w>'
                    word = col[0]
                    if len(word) > 30:
                        continue
                    if word in self.filtering_words:
                        # print(word)
                        continue
                    # subword_idx = create_composition_ngram(word, self.ngram_dic, index_only=True, limit_size=self.limit_size)
                    subword_idx = create_composition_ngram(word, self.ngram_dic, index_only=False, limit_size=self.limit_size)
                    # try:
                    #     print('word: {}'.format(word))
                    # except:
                    #     print('error')
                    # print('# of index: {}'.format(len(subword_idx)))
                    # print(subword_idx)
                    word_subidx_dic[i] = (word, subword_idx)
                    sum_idx += len(subword_idx)
                    word_count += 1
                    for index, _ in subword_idx:
                        idx_freq_dic[index] += 1

                    # check max length
                    # if maxlen < len(subword_idx):
                    #     maxlen = len(subword_idx)
                    #     print(maxlen)
                    #     print(word)
                    # assert maxlen > len(subword_idx)-1

                    # if len(subword_idx) < maxlen:
                    #     subword_idx_pad = subword_idx + [-1]*(maxlen-len(subword_idx))
                    # subword_idx_array = np.array(subword_idx_pad, dtype=np.int32)

                    # ref_vector = np.array(col[1:], dtype=np.float32)
                    # assert len(ref_vector) == dim

                    # y = np.array(0, dtype=np.int32)
                    # dataset.append((subword_idx_array, ref_vector, y))

                if self.test and i == 1000:
                    break

        print('average # of index: {}'.format(sum_idx/float(word_count)))
        print('\nlen(dataset) =', len(dataset), flush=True)

        fo = open('tmp','w')
        for i in range(vocab_size):
            try:
                word = word_subidx_dic[i][0]
                subword_idx = word_subidx_dic[i][1]
                fo.write('WORD: {},   # OF INDEX: {}\n'.format(word, len(subword_idx)))
            except:
                continue
            for sub_idx, token in subword_idx:
                try:
                    freq = idx_freq_dic[sub_idx]
                except:
                    freq = 0
                try:
                    fo.write('{} (id:{}, freq: {}), '.format(token, sub_idx, freq))
                except:
                    pass
            fo.write('\n')
        fo.write('\n')
        fo.write('average # of index: {}\n'.format(sum_idx/float(word_count)))
        fo.close()

        exit()
        return dataset
