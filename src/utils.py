# coding: utf-8
import sys
import numpy as np
import six
import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, variable
import chainer.functions as F
import chainer.links as L
import codecs
import json


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
                       'test',\
                       'gpu',\
                       'test_word_file_path'
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
                    # 
                    # TODO: handle this pattern
                    # 
                    # keep pre_args value into new Args
                    setattr(self, k, v)
                    # pass
            else:
                setattr(self, k, v)

def set_settings(args, settings):
    new_args = Args(args, settings)
    return new_args

def load_settings(args, model_path):
    return json.load(open(model_path+"settings.json"))

# basically this is same as the one on chainer's repo.
# I added padding option (padding=0) to be always true
# def concat_examples(batch, device=None, padding=None):
def concat_examples(batch, device=None, padding=0):
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    if len(batch) == 0:
        raise ValueError('batch is empty')

    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            if i == len(first_elem)-1:
                # for hinge loss function
                concat = _concat_arrays([example[i] for example in batch], padding[i])
                concat = np.reshape(concat, (len(batch), ))
            else:
                concat = _concat_arrays([example[i] for example in batch], padding[i])
            result.append(to_device(concat))
        return tuple(result)

    elif isinstance(first_elem, dict):
        print("挙動を理解していないのでこちらに入ったらError")
        assert False 
        exit()
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(_concat_arrays(
                [example[key] for example in batch], padding[key]))
        return result

def _concat_arrays(arrays, padding):
    if padding is not None:
        return _concat_arrays_with_padding(arrays, padding)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        # batch内で系列長が異なる時エラー
        return xp.concatenate([array[None] for array in arrays])
        # return xp.concatenate(arrays, axis=0)

def _concat_arrays_with_padding(arrays, padding):
    shape = np.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if np.any(shape != array.shape):
            np.maximum(shape, array.shape, shape)
    shape = tuple(np.insert(shape, 0, len(arrays)))

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        result = xp.full(shape, padding, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src

    return result

def get_total_line(path, test):
    if not(test):
        total_line = 0
        print('get # of lines', flush=True)
        with codecs.open(path, "r", 'utf-8', errors='replace') as input_data:
            for _ in input_data:
                total_line += 1
        print('done', flush=True)
        print('# of lines = {}'.format(total_line), flush=True)
    else:
        total_line = 1000
        print('# of lines = {}'.format(total_line), flush=True)

    return total_line
