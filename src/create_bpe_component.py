# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import argparse
import copy
import codecs

def merge(char_list, char_idx):
    return char_list[:char_idx] + [''.join(char_list[char_idx:char_idx+2])] + char_list[char_idx+2:]

def create_composition_bottom_up(word, merge_dic, index_only=False, limit_size=0, filter_flag=False):
    merge_list =[]
    char_list = ['^']
    pre_char_list = []
    if index_only:
        merge_list.append(merge_dic['^'][2])
    else:
        merge_list.append(merge_dic['^'])
    for char in word:
        char_list.append(char)
        try:
            if index_only:
                merge_list.append(merge_dic[char][2])
            else:
                merge_list.append(merge_dic[char])
        except:
            pass
    # char_list[-1] += '</w>'
    char_list.append('</w>')
    if index_only:
        merge_list.append(merge_dic['</w>'][2])
    else:
        merge_list.append(merge_dic['</w>'])
        
    while(len(char_list) != 1):
        min_merge_idx = 9999999999
        for i in range(len(char_list)-1):
            bi_gram = ''.join(c for c in char_list[i:i+2])
            try:
                merge_flag = True
                merge_element = merge_dic[bi_gram]
                merge_idx = merge_element[2]
            except:
                continue
            if min_merge_idx > merge_idx:
                min_merge_idx = merge_idx
                min_merge_element = merge_element
                char_idx = i

        if min_merge_idx == 9999999999:
            break
        pre_char_list = copy.deepcopy(char_list)
        char_list = merge(char_list, char_idx)
        if filter_flag:
            if merge_element[0].startswith('^') and merge_element[1].endswith('</w>'):
                continue 
        if limit_size > 0 and limit_size-1 < min_merge_idx:
            pass
        else:
            if index_only:
                merge_list.append(min_merge_element[2])
            else:
                merge_list.append(min_merge_element)
    return merge_list, char_list, pre_char_list
    
def set_base_merge(merge_dic, upper=True):
    if upper:
        base_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:\'-_/'
    else:    
        base_char = 'abcdefghijklmnopqrstuvwxyz0123456789.,:\'-_/'
    for i, char in enumerate(base_char):
        merge_dic[char] = [char, '', i]
        merge_dic[char+'</w>'] = [char, '</w>', i+len(base_char)]
    merge_dic['^'] = ['^', '', len(merge_dic)]
    merge_dic['</w>'] = ['</w>', '', len(merge_dic)]

    return merge_dic


def load_codecs(codecs_path, unique=False, upper=False):
    print('load codecs file', file=sys.stderr)
    merge_dic = {}
    merge_dic = set_base_merge(merge_dic, upper=upper)
    initial_size = len(merge_dic)
    for i, line in enumerate(open(codecs_path)):
        if i == 0:
            continue
        line = line.strip()
        merged = line.replace(' ','')
        if unique and merged.startswith('^') and merged.endswith('</w>'):
            continue
        col = line.strip().split()
        col.append(len(merge_dic))
        merge_dic[merged] = col
    print('done', file=sys.stderr)
    return merge_dic

def main(args):
    merge_dic = load_codecs(args.codecs_path)
    # merge_dic = load_codecs(args.codecs_path, unique=True)
    # word = 'ultimate'
    # print('create_composition')
    # merge_list, char_list, pre_char_list = create_composition_bottom_up(word, merge_dic)
    # print(merge_list)
    # print('done')
    # create_composition_bottom_up('ultimate', merge_dic,index_only=False, limit_size= 999999)

    for i, line in enumerate(codecs.open(args.vocab_path, "r", 'utf-8', errors='replace')):
        word = line.strip()
        merge_list, char_list, pre_char_list = create_composition_bottom_up(word, merge_dic)
        if len(char_list) == 1:
            print('\t'.join([sub for sub in pre_char_list]))
        else:
            print('\t'.join([sub for sub in char_list]))

    # from IPython.core.debugger import Pdb; Pdb().set_trace()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-arg1', action="store_true")
    parser.add_argument('--codecs_path', type=str)
    parser.add_argument('--vocab_path', type=str)
    args = parser.parse_args()
    main(args)











