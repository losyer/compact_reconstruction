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
    parser.add_argument('--gpu', dest='gpu', type=int,default=-1, help='negative value indicates CPU')

    # training parameter
    parser.add_argument('--epoch', dest='epoch', type=int,default=300, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,default=200, help='minibatch size')
    parser.add_argument('--snapshot_interval', type=int, default=50, help='')
    parser.add_argument('--snapshot_mintriger', action='store_true')
    parser.add_argument('--maxlen', type=int, default=-1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='')

    # training flag
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.add_argument('--raiden', action='store_true')
    parser.add_argument('--unique_false', action='store_false')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--hashed_idx', action='store_true')

    parser.add_argument('--load_embedding', action='store_true', help='')
    parser.add_argument('--load_parameter', action= 'store_true', help='')
    parser.add_argument('--load_snapshot', action='store_true', help='')

    parser.add_argument('--type_id', type=int, default=0, help='')
    parser.add_argument('--codecs_type', type=int, default=0, help='')
    parser.add_argument('--op_type', type=int, default=0, help='')
    parser.add_argument('--network_type', type=int, default=0, help='')
    # parser.add_argument('--subword_type', type=int, default=0, help='')
    parser.add_argument('--subword_type', type=int, default=4, help='')
    parser.add_argument('--z_type', type=int, default=0, help='')

    # other flag or id
    parser.add_argument('--job_id', type=str, default='-1', help='')
    parser.add_argument('--skip_create_dataset', action='store_true', help='')

    # model parameter
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=300, help='# of layer')
    parser.add_argument('--limit_size', type=int, default=10000, help='')
    parser.add_argument('--cnn_ksize', dest='cnn_ksize', type=int, default=3, help='')
    parser.add_argument('--bucket_size', type=int, default=10000, help='')
    parser.add_argument('--n_max', type=int, default=30, help='')
    parser.add_argument('--n_min', type=int, default=3, help='')
    parser.add_argument('--multi_hash', type=str, default='', help='')

    # data path
    parser.add_argument('--ref_vec_path', type=str, default="")
    parser.add_argument('--codecs_path', type=str, default="")
    parser.add_argument('--model_path', dest='model_path', type=str, default='')
    parser.add_argument('--result_dir', type=str, default="")
    parser.add_argument('--pretrained_subword_vector', type=str)
    parser.add_argument('--filtering_words_path', type=str, default="")
    parser.add_argument('--freq_path', type=str, default="")
    parser.add_argument('--pos_to_subword_path', type=str, default="")
    parser.add_argument('--bpe_codecs_path', type=str, default="")
    parser.add_argument('--test_word_file_path', type=str, default="")
    parser.add_argument('--test_words_pos_to_subword_path', type=str, default="")


    
    parser.add_argument('--output', type=str, default="")

    args = parser.parse_args()
    main(args)

