# coding: utf-8
import os, codecs, argparse, json, sys
HOME = os.getenv("HOME")
os.environ["CHAINER_SEED"] = "1"
import chainer
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import cuda, serializers
from chainer.dataset import concat_examples
import random
import numpy as np
random.seed(0)
np.random.seed(0)
chainer.config.cudnn_deterministic = True

# from net import Network, Network_cnn_inference
from net import Network, Network_cnn_new, Network_sum, Network_sep_kvq
from data_processor import DataProcessor
from datetime import datetime

class Args(object):
    def __init__(self, pre_args, settings):
        update_args = ['unique_false',\
                       'hashed_idx',\
                       'type_id',\
                       'codecs_type', \
                       'network_type',\
                       'subword_type',
                       'embed_dim',\
                       'limit_size',\
                       'cnn_ksize',\
                       'bucket_size',\
                       'n_max',\
                       'n_min',\
                       'ref_vec_path',\
                       'codecs_path',\
                       'filtering_words_path',\
                       'freq_path',\
                       'pos_to_subword_path',\
                       'bpe_codecs_path',\
                       'multi_hash',\
                       'maxlen',\
                       'z_type',\
                       'test'
                       # 'test_word_file_path'
                       ]
        super(Args, self).__init__()
        self.pre_args = vars(pre_args)
        for k, v in self.pre_args.items():
            if k in update_args:
                # 
                # TODO: handle appropriate arguments
                # 
                try:
                    setattr(self, k, settings[k])
                except:
                    pass
            else:
                setattr(self, k, v)

def set_settings(args, settings):
    new_args = Args(args, settings)
    return new_args

def load_settings(args, model_path):
    return json.load(open(model_path+"settings.json"))

def model_setup(args, n_vocab_subword):
    if args.network_type == 0:
        nn = Network(args, n_vocab_subword)
    elif args.network_type == 1:
        nn = Network_cnn_new(args, n_vocab_subword)
    elif args.network_type == 2:
        nn = Network_sum(args, n_vocab_subword)
    elif args.network_type == 3:
        nn = Network_sep_kvq(args, n_vocab_subword)    
    else:
        print('error')
        exit()    
    model = L.Classifier(nn, lossfun=F.mean_squared_error)
    
    model.compute_accuracy = False
    # optimizer setup
    optimizer = O.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    return model, optimizer

def set_out_name(args):
    out_name = ''
    if args.filtering_words_path != '':
        out_name = out_name + '.filter_words'
    if args.test_word_file_path != '':
        out_name = out_name + '.add_test_words'

    return out_name

def main(args):

    # data setup
    model_path = '/'.join(args.model_path.split('/')[0:-1])+'/'
    model_epoch = args.model_path.split('/')[-1].split('_')[-1]
    print('model path = ',model_path, flush=True)

    settings = load_settings(args, model_path)
    args = set_settings(args, settings)
    print(json.dumps(vars(args), sort_keys=True, indent=4), flush=True)
    # print('test reqired')
    # exit()

    data_processor = DataProcessor(args)
    n_vocab_subword = data_processor.n_vocab_subword
    filtering_words = data_processor.filtering_words
    model, optimizer = model_setup(args, n_vocab_subword)
    

    print("loading snapshot ...", flush=True)
    serializers.load_npz(model_path +'model_epoch_{}'.format(model_epoch), model, path='updater/model:main/')
    print('loading snapshot ... done', flush=True)
    save_path = model_path+'embedding_epoch{}'.format(model_epoch)
    try:
        os.makedirs(save_path)
    except:
        pass

    if args.save_model:
        print('saving model ...', flush=True)
        serializers.save_npz(save_path+'/model.npz', model)
        print('saving model ... done', flush=True)
        exit()

    error_count = 0
    if not args.save_all_subword_vec:
        print('batch reconstruct', flush=True)
        print('prepare dataset', flush=True)
        data_processor.prepare_dataset()
        print('calculate vectors...', flush=True)
        inference_iter = chainer.iterators.SerialIterator(data_processor.train_data, args.batchsize, repeat=False, shuffle=False)


        if args.time:
            import time
            start = time.time()
        for i, batch in enumerate(inference_iter):
            if args.network_type == 3:
                subword_idx_k, subword_idx_v, subword_idx_q, ref, freq, y = concat_examples(batch, device=args.gpu)
                vec = model.predictor(subword_idx_k, subword_idx_v, subword_idx_q, ref, freq)
            else:
                subword_idx, ref, freq, y = concat_examples(batch, device=args.gpu)
                vec = model.predictor(subword_idx, ref, freq)
            if args.time:
                continue

            vec = cuda.to_cpu(vec.data)
            if i == 0:
                all_vec = vec
            else:
                all_vec = np.concatenate([all_vec, vec])
        if args.time:
            elapsed_time = time.time() - start
            print('elapsed_time: {} sec / word'.format(elapsed_time/len(data_processor.train_data)))
            exit()

        print('### len(all_vec) = {}'.format(len(all_vec)), flush=True)
        print('write embedding ... ', flush=True)
        out_name = set_out_name(args)
        print('save_path = {}'.format(save_path+'/embedding_reconstruct_original'+out_name+'.txt'))
        fo = codecs.open(save_path+'/embedding_reconstruct_original'+out_name+'.txt','w','utf-8')

        print('write embedding ... (referenced vectors)', flush=True)
        for i, line in enumerate(codecs.open(args.ref_vec_path, "r", 'utf-8', errors='replace')):
            # 
            # かきこむ単語をlistとしてもってくる
            # from IPython.core.debugger import Pdb; Pdb().set_trace()
            # 
            if i == 0:
                col = line.strip().split()
                vocab_size = int(col[0])
                dim = int(col[1])
                if args.test:
                    # fo.write('{} {}\n'.format(1000+len(data_processor.added_test_words), dim))
                    fo.write('{} {}\n'.format(len(all_vec), dim))
                else:
                    # fo.write('{} {}\n'.format(vocab_size+len(data_processor.added_test_words), dim))
                    fo.write('{} {}\n'.format(len(all_vec), dim))
            else:
                if args.skip_create_dataset:
                    continue
                try:
                    word = line.strip().split()[0]
                except:
                    print('Warning: error line: {} ...'.format(line[:100]), flush=True)
                    error_count += 1
                    word = 'ERROR_TOKEN_{}'.format(error_count)
                fo.write('{} {}\n'.format(word,' '.join(['{:.6f}'.format(float(v)) for v in all_vec[i-1]])))

            if args.test and i == 1000:
                break

        if args.test_word_file_path != '':
            print('write embedding ... (test word vectors)', flush=True)
            for j, word in enumerate(data_processor.added_test_words):
                if args.skip_create_dataset:
                    fo.write('{} {}\n'.format(word,' '.join(['{:.6f}'.format(float(v)) for v in all_vec[j]])))
                else:
                    fo.write('{} {}\n'.format(word,' '.join(['{:.6f}'.format(float(v)) for v in all_vec[i-1+j+1]])))

        fo.close()
    else:
        links = [link for link in model.predictor.links()]
        for l in links:
            if l.name == 'predictor':
                continue
            # segment_type_num =  int(l.name[-1])
            fo = open(save_path+'/embedding','w')
            vocab_size = len(l.W.data)
            dim = len(l.W.data[0])
            fo.write('{} {}\n'.format(vocab_size, dim))
            vocab_inv = {v[2]:k for k, v in merge_dic.items()}

            # tempral handling
            # vocab_inv[0] = 'single_character'
            for j, vector in enumerate(l.W.data):
                word = vocab_inv[j]
                fo.write('{} {}\n'.format(word,' '.join(['{:.6f}'.format(float(v)) for v in vector])))
            fo.close()

    print('done')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int,default=-1, help='negative value indicates CPU')

    # training parameter
    parser.add_argument('--epoch', dest='epoch', type=int,default=5, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,default=2000, help='learning minibatch size')
    parser.add_argument('--maxlen', type=int, default=-1, help='')

    # training flag
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.add_argument('--unique_false', action='store_false')
    parser.add_argument('--hashed_idx', action='store_true')

    parser.add_argument('--type_id', type=int, default=0, help='')
    parser.add_argument('--codecs_type', type=int, default=0, help='')
    parser.add_argument('--network_type', type=int, default=0, help='')
    parser.add_argument('--subword_type', type=int, default=0, help='')
    parser.add_argument('--z_type', type=int, default=0, help='')

    # save flag
    parser.add_argument('--unk_random', action='store_true', help='')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--save_all_subword_vec', action='store_true', help='')
    # parser.add_argument('--normalize', action='store_true', help='')
    # parser.add_argument('--batch_reconstruct', action='store_true')

    # other flag
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--skip_create_dataset', action='store_true')

    # model parameter
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=300, help='# of layer')
    parser.add_argument('--limit_size', type=int, default=0, help='')
    parser.add_argument('--cnn_ksize', dest='cnn_ksize', type=int, default=3, help='')
    parser.add_argument('--bucket_size', type=int, default=1000000, help='')
    parser.add_argument('--n_max', type=int, default=30, help='')
    parser.add_argument('--n_min', type=int, default=1, help='')
    parser.add_argument('--multi_hash', type=str, default='', help='')

    # data path
    parser.add_argument('--ref_vec_path', type=str, default="")
    parser.add_argument('--codecs_path', type=str, default="")
    parser.add_argument('--model_path', dest='model_path', type=str, default="")
    parser.add_argument('--test_word_file_path', type=str, default="")
    parser.add_argument('--filtering_words_path', type=str, default="")
    parser.add_argument('--freq_path', type=str, default="")
    parser.add_argument('--pos_to_subword_path', type=str, default="")
    parser.add_argument('--test_words_pos_to_subword_path', type=str, default="")
    parser.add_argument('--write_word_path', type=str, default="")
    parser.add_argument('--bpe_codecs_path', type=str, default="")

    args = parser.parse_args()
    main(args)

