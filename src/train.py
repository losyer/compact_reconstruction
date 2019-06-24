# coding: utf-8
import json, sys, argparse, os 
from datetime import datetime
HOME = os.getenv("HOME")
os.environ["CHAINER_SEED"] = "1"
import chainer
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import cuda, serializers
import random
import numpy as np
random.seed(0)
np.random.seed(0)
chainer.config.cudnn_deterministic = True

from net import Network, Network_sum, Network_cnn_new, Network_sep_kvq
from data_processor import DataProcessor
from utils import Args, set_settings, load_settings

def set_result_dest(args):
    start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    if args.test:
        start_time = "test_" + start_time
    if args.network_type == 0:
        type_name = "kvq"
    elif args.network_type == 1:
        type_name = "cnn"
    elif args.network_type == 2:
        type_name = "sum"
    elif args.network_type == 3:
        type_name = "sep_kvq"
    else:
        print('error')
        exit()
    result_dest = args.result_dir+f"/{type_name}/"+start_time
    return result_dest

def initial_setup(args):
    # setup result directory
    result_dest = set_result_dest(args)
    result_abs_dest = os.path.abspath(result_dest)
    os.makedirs(result_dest)
    print("result dest: "+result_dest)

    # dump setting file
    with open(os.path.join(result_abs_dest, "settings.json"), "w") as fo:
        fo.write(json.dumps(vars(args), sort_keys=True, indent=4))
    print(json.dumps(vars(args), sort_keys=True, indent=4), flush=True)
    
    return result_dest

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
    if args.load_embedding:
        nn.load_embeddings(args, vocab_subword)

    model = L.Classifier(nn, lossfun=F.hinge)
    model.compute_accuracy = False
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = O.Adam(alpha=args.lr)
    optimizer.setup(model)

    return model, optimizer

def main(args):

    # initial setup
    if args.load_snapshot and args.model_path != '':
        print("load pre-settings")
        model_path = '/'.join(args.model_path.split('/')[0:-1])+'/'
        settings = load_settings(args, model_path)
        args = set_settings(args, settings)
        print('done')
    result_dest = initial_setup(args)

    # data setup    
    data_processor = DataProcessor(args)
    data_processor.prepare_dataset()
    n_vocab_subword = data_processor.n_vocab_subword

    # model setup
    model, optimizer = model_setup(args, n_vocab_subword)

    # iterator, updater and trainer setup
    train_iter = chainer.iterators.SerialIterator(data_processor.train_data, args.batchsize)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dest)

    model_path = '/'.join(args.model_path.split('/')[0:-1])+'/'
    model_epoch = args.model_path.split('/')[-1].split('_')[-1]
    print('model path = ',model_path)

    if args.load_snapshot:
        print("loading snapshot...")
        serializers.load_npz(model_path +'model_epoch_{}'.format(model_epoch), trainer)
        print('done')

    if args.load_parameter:
        print("loading parameter...")
        p1 = model.predictor.embed_subword
        serializers.load_npz(model_path+ "embed_subword.npz", p1)
        print('done')

    # Evaluation setup
    # # Log reporter setup
    trainer.extend(extensions.LogReport(log_name='log'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss_dev', 'validation/main/loss_test']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # if not args.test:
    if args.snapshot_mintriger:
        trainer.extend(extensions.snapshot(filename='model_epoch_{.updater.epoch}'),
                trigger=chainer.training.triggers.MinValueTrigger('main/loss'))
    else:
        trainer.extend(extensions.snapshot(filename='model_epoch_{.updater.epoch}'),
                      trigger=(args.snapshot_interval, 'epoch'))
    trainer.run()

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
    parser.add_argument('--network_type', type=int, default=0, help='')
    parser.add_argument('--subword_type', type=int, default=0, help='')
    parser.add_argument('--z_type', type=int, default=0, help='')

    # other flag or id
    # parser.add_argument('--job_id', type=str, default='-1', help='')
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
    
    args = parser.parse_args()
    main(args)

