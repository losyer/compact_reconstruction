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
from net import Network, Network_cnn_new
from data_processor import DataProcessor
from datetime import datetime
from create_composition import create_composition_ngram, load_codecs

def model_setup(args, n_vocab_subword):
    if args.cnn:
        # nn = Network_cnn_inference(args, n_vocab_subword)
        nn = Network_cnn_new(args, n_vocab_subword)
    else:
        nn = Network(args, n_vocab_subword)
    model = L.Classifier(nn, lossfun=F.mean_squared_error)
    
    model.compute_accuracy = False
    # optimizer setup
    optimizer = O.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    return model, optimizer

def main(args):

    print(json.dumps(vars(args), sort_keys=True, indent=4), flush=True)
    if args.inference:
        print('error')
        exit()

    # data setup
    data_processor = DataProcessor(args)
    # data_processor.prepare_dataset()
    n_vocab_subword = data_processor.n_vocab_subword
    ngram_dic = data_processor.ngram_dic
    model, optimizer = model_setup(args, n_vocab_subword)

    model_path = '/'.join(args.model_path.split('/')[0:-1])+'/'
    model_epoch = args.model_path.split('/')[-1].split('_')[-1]
    print('model path = ',model_path)

    print("loading snapshot...")
    serializers.load_npz(model_path +'model_epoch_{}'.format(model_epoch), model, path='updater/model:main/')
    print('done')

    test_set_words = set()
    for line in open(args.test_words_file,'r'):
        word = line.strip()
        test_set_words.add(word)

    dataset = []
    maxlen = 135
    print('prepare data', flush=True)
    done_words=[]
    # fo = codecs.open('tmp.vec', "w", 'utf-8', errors='replace')
    with codecs.open(args.ref_vec_path, "r", 'utf-8', errors='replace') as input_data:
        for i, line in enumerate(input_data):
            col = line.strip().split()
            if i == 0:
                vocab_size, dim = int(col[0]), int(col[1])
                continue
            else:
                word = col[0]   
            if word not in test_set_words:
                continue
            done_words.append(word)
            subword_idx = create_composition_ngram(word, ngram_dic, index_only=True, limit_size=args.limit_size)
            # check max length
            if len(subword_idx) > maxlen:
                continue
            if len(subword_idx) <= maxlen:
                subword_idx_pad = subword_idx + [-1]*(maxlen-len(subword_idx))
            subword_idx_array = np.array(subword_idx_pad, dtype=np.int32)
            ref_vector = np.array(col[1:], dtype=np.float32)
            dataset.append((subword_idx_array, ref_vector))
            # dataset.append((word, ref_vector))

    # fo.write('{} {}\n'.format(len(dataset), 300))
    # for word, vec in dataset:
    #     fo.write('{} {}\n'.format(word,' '.join(['{:.6f}'.format(float(v)) for v in vec])))
    # fo.close()
    # exit()

            # if len(done_words) == len(test_set_words):
                # hairanai
                # break
            # if len(done_words) == 3000:
            #     break

    print('len(dataset) = ',len(dataset))
    print('calculate loss...', flush=True)
    inference_iter = chainer.iterators.SerialIterator(dataset, args.batchsize, repeat=False, shuffle=False)
    for i, batch in enumerate(inference_iter):
        subword_idx, ref = concat_examples(batch, device=args.gpu)
        loss = model.predictor(subword_idx, ref)
        if i == 0:
            loss_sum = sum(loss.data)
        else:
            loss_sum += sum(loss.data)
        sys.stderr.write('\rcalculating: %d' % ((i+1)*args.batchsize))
    print('')
    print('average loss: ', 1.0 - loss_sum/len(dataset))
    print('done')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,default=-1, help='negative value indicates CPU')

    # training parameter
    parser.add_argument('--epoch', dest='epoch', type=int,default=5, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,default=2000, help='learning minibatch size')
    parser.add_argument('--cnn_ksize', dest='cnn_ksize', type=int, default=3, help='')

    # training flag
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.add_argument('--raiden', action='store_true', help='')
    parser.add_argument('--unk_random', action='store_true', help='')
    parser.add_argument('--unique', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--apply_freq', action='store_true')
    parser.add_argument('--type_id', type=int, default=0, help='')
    parser.add_argument('--codecs_type', type=int, default=0, help='')

    # other flag
    parser.add_argument('--reconstruct_original', action='store_true', help='')

    # model parameter
    parser.add_argument('--limit_size', type=int, default=0, help='')
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=300, help='# of layer')

    # data path
    parser.add_argument('--ref_vec_path', type=str, default="")
    parser.add_argument('--codecs_path', type=str, default="")
    parser.add_argument('--model_path', dest='model_path', type=str, default='')
    parser.add_argument('--test_word_file_path', type=str, default="")
    parser.add_argument('--filtering_words_path', type=str, default="")
    parser.add_argument('--test_words_file', type=str, default="")

    args = parser.parse_args()
    main(args)

