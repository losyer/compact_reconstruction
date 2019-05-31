# -*- coding: utf-8 -*-
import argparse
import codecs

def hash( str , prime=0x01000193):
    hval = 0x811c9dc5
    # fnv_32_prime = 0x01000193
    fnv_32_prime = prime
    uint32_max = 2 ** 32
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * fnv_32_prime) % uint32_max
    return hval

def merge(char_list, char_idx):
    return char_list[:char_idx] + [''.join(char_list[char_idx:char_idx+2])] + char_list[char_idx+2:]

def create_composition_bottom_up_with_hash(word, merge_dic, index_only=False, limit_size=0, filter_flag=False):
    merge_list =[]
    char_list = ['^']
    if index_only:
        merge_list.append(hash('BPE-^'))
    else:
        merge_list.append([hash('BPE-^'),'BPE-^'])
    for char in word:
        char_list.append(char)
        try:
            if index_only:
                merge_list.append(hash(char))
            else:
                merge_list.append([hash(char), 'BPE-'+char])
        except:
            pass
    # char_list[-1] += '</w>'
    char_list.append('</w>')
    if index_only:
        merge_list.append(hash('BPE-</w>'))
    else:
        merge_list.append([hash('BPE-</w>'), 'BPE-</w>'])
        
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
        char_list = merge(char_list, char_idx)
        if filter_flag:
            if merge_element[0].startswith('^') and merge_element[1].endswith('</w>'):
                continue 
        if limit_size > 0 and limit_size-1 < min_merge_idx:
            pass
        else:
            sub_left = 'BPE-' + min_merge_element[0]
            sub_right = 'BPE-' + min_merge_element[1]
            subword = sub_left+sub_right
            if index_only:
                merge_list.append(hash(subword))
            else:
                merge_list.append([hash(subword), subword])
    return merge_list
    
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


def load_bpe_codecs(codecs_path, unique=False, upper=True):
    print('load codecs file')
    merge_dic = {}
    merge_dic = set_base_merge(merge_dic, upper=upper)
    initial_size = len(merge_dic)
    for i, line in enumerate(codecs.open(codecs_path, "r", 'utf-8', errors='replace')):
        if i == 0:
            continue
        line = line.strip()
        merged = line.replace(' ','')
        if unique and merged.startswith('^') and merged.endswith('</w>'):
            continue
        col = line.strip().split()
        col.append(len(merge_dic))
        merge_dic[merged] = col
    print('done')
    return merge_dic

def main(args):
    merge_dic = load_bpe_codecs(args.codecs_path)
    # merge_dic = load_codecs(args.codecs_path, unique=True)
    word = 'ultimate'
    print('create_composition')
    merge_list = create_composition_bottom_up_with_hash(word, merge_dic)
    print(merge_list)
    print('done')

     # create_composition_bottom_up('ultimate', merge_dic,index_only=False, limit_size= 999999)
    # from IPython.core.debugger import Pdb; Pdb().set_trace()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-arg1', action="store_true")
    parser.add_argument('--codecs_path', type=str)
    args = parser.parse_args()
    main(args)





