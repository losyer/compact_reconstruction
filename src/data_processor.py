# -*- coding: utf-8 -*-
import sys, json, os, codecs
import numpy as np
from collections import defaultdict
from itertools import groupby
from chainer import cuda
from create_composition import load_codecs, create_composition_ngram, create_composition_ngram_with_hash
# from create_composition import load_codecs, load_codecs_skip_ngram, load_pos_to_subwordlist
# from create_composition import create_composition_ngram, create_composition_ngram_with_hash, create_composition_bpe_with_hash, create_composition_skip_ngram
from create_composition import hash
from utils import get_total_line

class DataProcessor(object):

    def __init__(self, args):
        self.args = args
        self.ref_vec_path = args.ref_vec_path
        self.test = args.test
        self.limit_size = args.limit_size
        if self.args.freq_path != "":
            self.create_word_to_freq_dic()
        if args.codecs_path != '':
            self.ngram_dic = self.create_ngram_dic()
        else:
            self.ngram_dic = None
        self.set_n_vocab_subword()
        self.set_maxlen()
        self.filtering_words = self.load_filtering_words(args.filtering_words_path)
        self.separate_kvq = True if self.args.network_type == 3 else False
        self.loaded_words_set = set()
        self.added_test_words = []
        self.multi_hash = args.multi_hash
        self.skip_create_dataset = args.skip_create_dataset
        self.total_line = None

    def get_freq(self, word):
        try:
            return self.word_to_freq_dic[word]
        except:
            return 2

    def create_word_to_freq_dic(self):
        print("create word to frequency dictionary ...", flush=True)
        self.word_to_freq_dic = {}
        for line in codecs.open(self.args.freq_path, "r", 'utf-8', errors='replace'):
            col = line.strip().split("\t")
            assert len(col) == 2
            word, freq = col[0], int(col[1])
            self.word_to_freq_dic[word] = freq
        print("create word to frequency dictionary ... done", flush=True)

    def create_ngram_dic(self):
        if self.args.subword_type == 0 or self.args.subword_type == 4:
            # TODO: use appropriate number
            return load_codecs(self.args.codecs_path, self.args.limit_size,\
                               self.args.unique_false, self.args.codecs_type)
        else:
            print('error 2')
            exit()

    def set_n_vocab_subword(self):
        if self.args.hashed_idx:
            self.n_vocab_subword = self.args.bucket_size
        elif self.limit_size > 0:
            if self.args.network_type == 3:
                self.n_vocab_subword = int(self.limit_size*3)
            else:
                self.n_vocab_subword = self.limit_size
        else:
            print('error 3')
            exit()
        print('n_vocab_subword = ', self.n_vocab_subword, flush=True)

    def set_maxlen(self):
        if self.args.maxlen != -1:
            self.maxlen = self.args.maxlen
        elif self.args.subword_type == 0 or self.args.subword_type == 4:
            self.maxlen = 135
        else:
            print('error 4')
            exit()

    def load_filtering_words(self, path):
        if path == '':
            return set()
        else:
            print('loading filtering words ...', flush=True)
            words = set()
            for line in codecs.open(path, "r", 'utf-8', errors='replace'):
                word = line.strip()
                words.add(word)
            print('loading filtering words ... done', flush=True)
            return words

    def prepare_dataset(self):
        self.train_data = self.load_dataset()
        if self.test:
            print("\ntiny dataset for quick test...", flush=True)
            print('length of dataset =', len(self.train_data), flush=True)

        if self.args.inference and self.args.test_word_file_path != "":
            self.train_data = self.add_unk_words_for_test(self.train_data)

    def load_dataset(self):
        dataset = []
        self.total_line = get_total_line(path=self.ref_vec_path, test=self.test)
        # self.create_freq_dic()

        print("create dataset ...", flush=True)
        with codecs.open(self.ref_vec_path, "r", 'utf-8', errors='replace') as input_data:
            for i, line in enumerate(input_data):
                
                if i % int(self.total_line/10) == 0:
                    print('{} % done'.format(round(i / (self.total_line/100))), flush=True)
                # col = line.strip().split()
                if i == 0:
                    col = line.strip('\n').split()
                    vocab_size, dim = int(col[0]), int(col[1])
                    continue
                # col = line.strip().rsplit(' ', dim)
                col = line.rstrip(' \n').rsplit(' ', dim)
                word = col[0]
                if self.args.inference:
                    if self.skip_create_dataset:
                        break
                    self.loaded_words_set.add(word)
                    if word in self.filtering_words or len(col) != dim+1:
                        pass
                    if len(word) > 30:
                        # new_subword_idx = [-1]*self.maxlen
                        new_subword_idx = []
                    else:
                        subword_idx = self.get_subword_idx(word, i-1)
                        new_subword_idx = self.check_subword_idx(subword_idx)
                else:
                    if word in self.filtering_words or len(word) > 30 or len(col) != dim+1:
                        continue
                    self.loaded_words_set.add(word)
                    subword_idx = self.get_subword_idx(word, i-1)
                    new_subword_idx = self.check_subword_idx(subword_idx)
                    if len(new_subword_idx) == 0:
                        continue
                assert len(new_subword_idx) <= self.maxlen

                ref = [None] if self.args.inference else col[1:]
                ref_vector = np.array(ref, dtype=np.float32)
                y = np.array(0, dtype=np.int32)
                freq = self.get_freq(word) if self.args.freq_path!="" else 1
                freq_array = np.array(freq, dtype=np.float32)
                dataset = self.set_dataset(dataset, new_subword_idx, ref_vector, freq_array, y)

                if self.test and len(dataset) == 1000:
                    break
        print("create dataset ... done", flush=True)
        print('len(dataset) =', len(dataset), flush=True)
        return dataset

    def get_subword_idx(self, word, pos, limit_false=True, unk=False):

        '''
        methods subword_type hashed_idx
        --------------------------------
        SUM-F          0        False
        SUM-H          0        True 
        KVQ-H          0        True 
        SUM-FH         4        True 
        KVQ-FH         4        True 
        '''

        index_only = False if self.separate_kvq else True
        if self.args.hashed_idx:
            if self.args.subword_type == 0:
                subword_idx = create_composition_ngram_with_hash(word, self.args.n_max, self.args.n_min,\
                                                             index_only=index_only, multi_hash=self.multi_hash)
            elif self.args.subword_type == 4: # TODO: use appropriate number
                subword_idx = create_composition_ngram_with_hash(word, self.args.n_max, self.args.n_min,\
                                   index_only=index_only, multi_hash=self.multi_hash, ngram_dic=self.ngram_dic)
            else:
                print('error 5')
                exit()
        else:
            if self.args.subword_type == 0:
                subword_idx = create_composition_ngram(word, self.ngram_dic, self.args.n_max,\
                                                       self.args.n_min, self.limit_size, index_only=index_only)
            else:
                print('error 6')
                exit()
        return subword_idx

    def check_subword_idx(self, subword_idx, unkwords=False):
        new_subword_idx_list = []
        if self.separate_kvq:
            for idx, subword in subword_idx:
                if self.args.hashed_idx:
                    idx = idx % self.args.bucket_size
                new_subword_idx_list.append((idx, subword))
        else:
            for idx in subword_idx:
                if self.args.hashed_idx:
                    idx = idx % self.args.bucket_size
                new_subword_idx_list.append(idx)
        if len(new_subword_idx_list) > self.maxlen:
            # continue
            new_subword_idx_list = new_subword_idx_list[:self.maxlen]
        return new_subword_idx_list

    def set_dataset(self, dataset, new_subword_idx, ref_vector, freq_array, y):
        if self.args.hashed_idx:
            if self.separate_kvq:
                subword_idx_k = [hashed_idx for hashed_idx, _ in new_subword_idx]
                subword_idx_v = [hash(subword, prime=0x01000194) % self.args.bucket_size for _, subword in new_subword_idx]
                subword_idx_q = [hash(subword, prime=0x01000195) % self.args.bucket_size for _, subword in new_subword_idx]
                subword_idx_array_k = self.pad_array(subword_idx_k)
                subword_idx_array_v = self.pad_array(subword_idx_v)
                subword_idx_array_q = self.pad_array(subword_idx_q)
                dataset.append((subword_idx_array_k, subword_idx_array_v, subword_idx_array_q, ref_vector, freq_array, y))
            else:
                subword_idx_array = self.pad_array(new_subword_idx)
                dataset.append((subword_idx_array, ref_vector, freq_array, y))
        else:
            if self.separate_kvq:
                subword_idx_k = [idx for idx, _ in new_subword_idx]
                subword_idx_v = [idx+int(self.n_vocab_subword/3) for idx, subword in new_subword_idx]
                subword_idx_q = [idx+int(self.n_vocab_subword/3*2) for idx, subword in new_subword_idx]
                subword_idx_array_k = self.pad_array(subword_idx_k)
                subword_idx_array_v = self.pad_array(subword_idx_v)
                subword_idx_array_q = self.pad_array(subword_idx_q)
                dataset.append((subword_idx_array_k, subword_idx_array_v, subword_idx_array_q, ref_vector, freq_array, y))
            else:
                subword_idx_array = self.pad_array(new_subword_idx)
                dataset.append((subword_idx_array, ref_vector, freq_array, y))

        return dataset

    def pad_array(self, idx_list):
        idx_pad = idx_list + [-1]*(self.maxlen-len(idx_list))
        idx_array = np.array(idx_pad, dtype=np.int32)
        return idx_array

    def add_unk_words_for_test(self, dataset):
        # not implemented
        # load test set words
        print('[add unknown words] ...', flush=True)
        test_words = []
        for line in codecs.open(self.args.test_word_file_path, "r", 'utf-8', errors='replace'):
            word = line.strip()
            test_words.append(word)
        test_words = set(test_words)
        for i, word in enumerate(test_words):
            if word not in self.loaded_words_set:
                self.added_test_words.append(word)
                if len(word) > 30:
                    new_subword_idx = []
                else:
                    subword_idx = self.get_subword_idx(word, i, unk=True)
                    new_subword_idx = self.check_subword_idx(subword_idx)
                assert len(new_subword_idx) <= self.maxlen

                ref = [None] if self.args.inference else col[1:]
                ref_vector = np.array(ref, dtype=np.float32)
                y = np.array(0, dtype=np.int32)
                freq = self.get_freq(word) if self.args.freq_path!="" else 1
                freq_array = np.array(freq, dtype=np.float32)
                dataset = self.set_dataset(dataset, new_subword_idx, ref_vector, freq_array, y)

        print('[add unknown words] len(dataset) =', len(dataset), flush=True)
        return dataset


    def prepare_data_for_inference(self):
        print("prepare_data_for_inference ...", flush=True)
        # self.total_line = get_total_line(path=self.args.oov_word_path, test=self.test)
        dataset = []
        with codecs.open(self.args.oov_word_path, "r", 'utf-8', errors='replace') as input_data:
            for i, line in enumerate(input_data):
                
                # if i % int(self.total_line/10) == 0:
                #     print('{} % done'.format(round(i / (self.total_line/100))), flush=True)
                word = line.strip()
                if len(word) > 30:
                    new_subword_idx = []
                else:
                    subword_idx = self.get_subword_idx(word, pos=None)
                    new_subword_idx = self.check_subword_idx(subword_idx)
                assert len(new_subword_idx) <= self.maxlen

                ref = [None]
                ref_vector = np.array(ref, dtype=np.float32)
                y = np.array(0, dtype=np.int32)
                freq = self.get_freq(word) if self.args.freq_path!="" else 1
                freq_array = np.array(freq, dtype=np.float32)
                dataset = self.set_dataset(dataset, new_subword_idx, ref_vector, freq_array, y)

        print("prepare_data_for_inference ... done", flush=True)
        print('len(data) =', len(dataset), flush=True)
        self.oov_data = dataset








