# -*- coding: utf-8 -*-
import sys
import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Chain
from chainer import Variable as V
RS = np.random.RandomState(0)

class Network(Chain):
    def __init__(self, args, n_vocab_subword, train=True):
        self.args = args
        self.train = train
        self.embed_dim = args.embed_dim
        self.n_vocab_subword = n_vocab_subword
        # self.lambda_value = args.lambda_value
        self.type_id = args.type_id
        self.inference = args.inference

        super(Network, self).__init__(
            # embed_ref=L.EmbedID(self.n_vocab_ref, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_ref, self.embed_dim))),
            # embed_subword=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
            embed_subword=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.01,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
            embed_subword_K=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.01,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
            embed_subword_V=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.01,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),            
        )

    def load_embeddings(self, args, vocab):
        embed = self.embed_subword
        vec_path = args.pretrained_subword_vector

        print("loading pretrained-vectors ...")
        with open(vec_path, "r") as fi:
            for i, line in enumerate(fi):
                if i == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0]
                if word in vocab:
                    value = line.strip().split(" ")[1::]
                    vec = self.xp.array(value, dtype=self.xp.float32)
                    embed.W.data[vocab[word]] = vec
        print("done")

    def __call__(self, subword_idx, ref_vector, freq):

        batchsize = len(subword_idx)
        if self.inference:
            #subword_vector = F.sum(self.embed_subword(subword_idx), axis=1)
            subword_vector = self.get_word_vec_attn(subword_idx)
            return subword_vector

        sq_loss = self.func_weighted_normalized_attn(subword_idx, ref_vector)
        #sq_loss = F.sum(F.squared_error(subword_vector, ref_vector), axis=1, keepdims=True)/self.embed_dim

        if self.args.freq_path != "":
            sq_loss = sq_loss*F.reshape(F.log(freq), (batchsize, 1))

        if self.type_id == 0:
            return 1.0-sq_loss

        if self.type_id == 1:
            ref_vector_sq0 = F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True))
            return 1.0-ref_vector_sq0*sq_loss

    def get_word_vec_attn(self, subword_idx):
        batchsize, seq_len = subword_idx.shape
        hdim = self.embed_subword.W.shape[1]
        sp  = (batchsize, hdim)
        sp2 = (batchsize, seq_len, hdim)
        subword_mask = self.xp.array(subword_idx, dtype=self.xp.float32)
        mask = self.xp.broadcast_to(self.xp.reshape(-(F.relu(-subword_mask).data)*10000.0, (batchsize, seq_len, 1)), sp2)
        
        sub_vector_Q = self.embed_subword(subword_idx)
        sub_vector_K = self.embed_subword_K(subword_idx)
        #sub_vector_Q = sub_vector_K
        sub_vector_V = self.embed_subword_V(subword_idx)        

        sub_vector_Q  = F.broadcast_to(F.sum(sub_vector_Q, axis=1, keepdims=True), sp2)
        #sub_vector_Q1 = F.broadcast_to(F.max(sub_vector_Q, axis=1, keepdims=True), sp2)
        #sub_vector_Q2 = F.broadcast_to(F.min(sub_vector_Q, axis=1, keepdims=True), sp2)        

        #zero_vec = self.xp.zeros(sub_vector_Q.shape, dtype=self.xp.float32)
        #(re1, im1) = F.fft((sub_vector_Q, zero_vec))
        #(re2, im2) = F.fft((sub_vector_K, zero_vec))
        #sub_vector_QK  = F.softmax(( F.ifft((re1*re2, im1*im2))[0] )*self.xp.sqrt(hdim) + mask, axis=1)

        #sub_vector_QK = F.softmax((F.tanh(sub_vector_Q) * F.tanh(sub_vector_K))*self.xp.sqrt(hdim) + mask, axis=1)
        sub_vector_QK = F.softmax((sub_vector_Q * sub_vector_K)*self.xp.sqrt(hdim) + mask, axis=1)        
        #sub_vector_QK1 = F.softmax((sub_vector_Q1 * sub_vector_K) + mask, axis=1)
        #sub_vector_QK2 = F.softmax((sub_vector_Q2 * sub_vector_K) + mask, axis=1)
        #sub_vector    = F.sum(sub_vector_QK * sub_vector_V + sub_vector_QK1 * sub_vector_V + sub_vector_QK2 * sub_vector_V, axis=1)

        sub_vector    = F.sum(sub_vector_QK * sub_vector_V , axis=1)

        #sub_vector    = F.sum(sub_vector_QK * (sub_vector_V + sub_vector_Q), axis=1)        
        
        #zero_vec = self.xp.zeros(sub_vector_QK.shape, dtype=self.xp.float32)
        #(re1, im1) = F.fft((sub_vector_QK, zero_vec))
        #(re2, im2) = F.fft((sub_vector_V, zero_vec))
        #sub_vector = F.sum(F.ifft((re1*re2, im1*im2))[0], axis=1)
        return sub_vector


    def func_weighted_normalized_attn(self, subword_idx, ref_vector):
        batchsize = len(subword_idx)
        sp = (batchsize, self.embed_subword.W.shape[1])
        sub_vector = self.get_word_vec_attn(subword_idx)

        sub_vector_sq  = F.broadcast_to(F.sqrt(F.sum(sub_vector * sub_vector, axis=1, keepdims=True)), sp)
        sub_vector = sub_vector / sub_vector_sq  # 単位vector化

        ref_vector_sq0 = F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True))
        ref_vector_sq  = F.broadcast_to(ref_vector_sq0, sp)
        # ref_vector_sq  = F.broadcast_to(F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True)), sp)
        ref_vector = ref_vector / ref_vector_sq  # 単位vector化

        sq_loss = F.sum(F.squared_error(sub_vector, ref_vector)/self.embed_dim, axis=1, keepdims=True)
        return sq_loss

    
class Network_sum(Chain):
    def __init__(self, args, n_vocab_subword, train=True):
        self.args = args
        self.train = train
        self.embed_dim = args.embed_dim
        self.n_vocab_subword = n_vocab_subword
        # self.lambda_value = args.lambda_value
        self.type_id = args.type_id
        self.inference = args.inference

        super(Network_sum, self).__init__(
            # embed_ref=L.EmbedID(self.n_vocab_ref, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_ref, self.embed_dim))),
            # embed_subword=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
            embed_subword=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.01,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
        )

    def load_embeddings(self, args, vocab):
        embed = self.embed_subword
        vec_path = args.pretrained_subword_vector

        print("loading pretrained-vectors ...")
        with open(vec_path, "r") as fi:
            for i, line in enumerate(fi):
                if i == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0]
                if word in vocab:
                    value = line.strip().split(" ")[1::]
                    vec = self.xp.array(value, dtype=self.xp.float32)
                    embed.W.data[vocab[word]] = vec
        print("done")

    def __call__(self, subword_idx, ref_vector, freq):

        batchsize = len(subword_idx)
        sub_vector = F.sum(self.embed_subword(subword_idx), axis=1)

        if self.inference:
            return sub_vector

        # 単位vector化
        sp = (batchsize, self.embed_subword.W.shape[1])
        sub_vector_sq  = F.broadcast_to(F.sqrt(F.sum(sub_vector * sub_vector, axis=1, keepdims=True)), sp)
        sub_vector = sub_vector / sub_vector_sq
        ref_vector_sq  = F.broadcast_to(F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True)), sp)
        ref_vector = ref_vector / ref_vector_sq

        sq_loss = F.sum(F.squared_error(sub_vector, ref_vector), axis=1, keepdims=True)/self.embed_dim
        if self.args.freq_path != "":
            sq_loss = sq_loss*F.reshape(F.log(freq), (batchsize, 1))

        if self.type_id == 0:
            return 1.0-sq_loss

        if self.type_id == 1:
            ref_vector_sq0 = F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True))
            return 1.0-ref_vector_sq0*sq_loss


class Network_cnn_new(Chain):
    def __init__(self, args, n_vocab_subword, train=True):
        self.train = train
        self.embed_dim = args.embed_dim
        self.n_vocab_subword = n_vocab_subword
        self.cnn_ksize = args.cnn_ksize
        self.inference = args.inference

        super(Network_cnn_new, self).__init__(
            # embed_ref=L.EmbedID(self.n_vocab_ref, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_ref, self.embed_dim))),
            conv_q=L.Convolution2D(in_channels=1, out_channels=self.embed_dim, ksize=(self.cnn_ksize, self.embed_dim)),
            embed_subword=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
        )

    def load_embeddings(self, args, vocab):
        embed = self.embed_subword
        vec_path = args.pretrained_subword_vector

        print("loading pretrained-vectors ...")
        with open(vec_path, "r") as fi:
            for i, line in enumerate(fi):
                if i == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0]
                if word in vocab:
                    value = line.strip().split(" ")[1::]
                    vec = self.xp.array(value, dtype=self.xp.float32)
                    embed.W.data[vocab[word]] = vec
        print("done")

    def __call__(self, subword_idx, ref_vector, freq):

        batchsize, seq_len = subword_idx.shape
        subword_embed = self.embed_subword(subword_idx)
        subword_conv = F.tanh(self.conv_q(F.reshape(subword_embed, (batchsize, 1, seq_len, self.embed_dim))))
        # avg_pool = F.average_pooling_2d(subword_conv, ksize=(subword_conv.shape[2],1))
        # avg_pool = F.reshape(avg_pool, (batchsize, self.embed_dim))
        subword_sum = F.sum(subword_conv, axis=2)
        subword_sum = F.reshape(subword_sum, (batchsize, self.embed_dim))

        if self.inference:
            return subword_sum

        sp = (batchsize, self.embed_subword.W.shape[1])
        ref_vector_sq0 = F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True))
        ref_vector_sq  = F.broadcast_to(ref_vector_sq0, sp)
        ref_vector_normalized = ref_vector / ref_vector_sq 

        sq_loss = F.reshape(F.sum(F.squared_error(subword_sum, ref_vector_normalized)/self.embed_dim, axis=1), (batchsize, 1))
        if self.args.freq_path != "":
            sq_loss = sq_loss*F.reshape(F.log(freq), (batchsize, 1))
        # sq_loss = F.reshape(F.sum(F.squared_error(avg_pool, ref_vector_normalized)/self.embed_dim, axis=1), (batchsize, 1))
        return 1-sq_loss

    
class Network_sep_kvq(Chain):
    def __init__(self, args, n_vocab_subword, train=True):
        self.args = args
        self.train = train
        self.embed_dim = args.embed_dim
        self.n_vocab_subword = n_vocab_subword
        # self.lambda_value = args.lambda_value
        self.type_id = args.type_id
        self.inference = args.inference

        super(Network_sep_kvq, self).__init__(
            # embed_ref=L.EmbedID(self.n_vocab_ref, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_ref, self.embed_dim))),
            # embed_subword=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
            embed_subword=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.01,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
            # embed_subword_K=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.01,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
            # embed_subword_V=L.EmbedID(self.n_vocab_subword, self.embed_dim, initialW=RS.normal(scale=0.01,size=(n_vocab_subword, self.embed_dim)), ignore_label=-1),
        )

    def load_embeddings(self, args, vocab):
        embed = self.embed_subword
        vec_path = args.pretrained_subword_vector

        print("loading pretrained-vectors ...")
        with open(vec_path, "r") as fi:
            for i, line in enumerate(fi):
                if i == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0]
                if word in vocab:
                    value = line.strip().split(" ")[1::]
                    vec = self.xp.array(value, dtype=self.xp.float32)
                    embed.W.data[vocab[word]] = vec
        print("done")

    def __call__(self, subword_idx_k, subword_idx_v, subword_idx_q, ref_vector, freq):

        batchsize = len(subword_idx_k)
        if self.inference:
            #subword_vector = F.sum(self.embed_subword(subword_idx), axis=1)
            subword_vector = self.get_word_vec_attn(subword_idx_k, subword_idx_v, subword_idx_q)
            return subword_vector

        sq_loss = self.func_weighted_normalized_attn(subword_idx_k, subword_idx_v, subword_idx_q, ref_vector)
        #sq_loss = F.sum(F.squared_error(subword_vector, ref_vector), axis=1, keepdims=True)/self.embed_dim

        if self.args.freq_path != "":
            sq_loss = sq_loss*F.reshape(F.log(freq), (batchsize, 1))

        if self.type_id == 0:
            return 1.0-sq_loss

        if self.type_id == 1:
            ref_vector_sq0 = F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True))
            return 1.0-ref_vector_sq0*sq_loss

    def get_word_vec_attn(self, subword_idx_k, subword_idx_v, subword_idx_q):
        batchsize, seq_len = subword_idx_k.shape
        hdim = self.embed_subword.W.shape[1]
        sp  = (batchsize, hdim)
        sp2 = (batchsize, seq_len, hdim)

        ####
        subword_mask = self.xp.array(subword_idx_k, dtype=self.xp.float32)
        mask = self.xp.broadcast_to(self.xp.reshape(-(F.relu(-subword_mask).data)*10000.0, (batchsize, seq_len, 1)), sp2)
        ####
        
        sub_vector_Q = self.embed_subword(subword_idx_q)
        sub_vector_K = self.embed_subword(subword_idx_k)
        #sub_vector_Q = sub_vector_K
        sub_vector_V = self.embed_subword(subword_idx_v)

        sub_vector_Q  = F.broadcast_to(F.sum(sub_vector_Q, axis=1, keepdims=True), sp2)
        #sub_vector_Q1 = F.broadcast_to(F.max(sub_vector_Q, axis=1, keepdims=True), sp2)
        #sub_vector_Q2 = F.broadcast_to(F.min(sub_vector_Q, axis=1, keepdims=True), sp2)        

        #zero_vec = self.xp.zeros(sub_vector_Q.shape, dtype=self.xp.float32)
        #(re1, im1) = F.fft((sub_vector_Q, zero_vec))
        #(re2, im2) = F.fft((sub_vector_K, zero_vec))
        #sub_vector_QK  = F.softmax(( F.ifft((re1*re2, im1*im2))[0] )*self.xp.sqrt(hdim) + mask, axis=1)

        #sub_vector_QK = F.softmax((F.tanh(sub_vector_Q) * F.tanh(sub_vector_K))*self.xp.sqrt(hdim) + mask, axis=1)

        if self.args.z_type == 0:
            sub_vector_QK = F.softmax((sub_vector_Q * sub_vector_K)*self.xp.sqrt(hdim) + mask, axis=1)
        elif self.args.z_type == 1:
            sub_vector_QK = F.softmax((sub_vector_Q * sub_vector_K) + mask, axis=1)
        elif self.args.z_type == 2:
            sub_vector_QK = F.softmax((sub_vector_Q * sub_vector_K)/self.xp.sqrt(hdim) + mask, axis=1)
        else:
            print('error')
            exit()
        #sub_vector_QK1 = F.softmax((sub_vector_Q1 * sub_vector_K) + mask, axis=1)
        #sub_vector_QK2 = F.softmax((sub_vector_Q2 * sub_vector_K) + mask, axis=1)
        #sub_vector    = F.sum(sub_vector_QK * sub_vector_V + sub_vector_QK1 * sub_vector_V + sub_vector_QK2 * sub_vector_V, axis=1)

        sub_vector    = F.sum(sub_vector_QK * sub_vector_V , axis=1)

        #sub_vector    = F.sum(sub_vector_QK * (sub_vector_V + sub_vector_Q), axis=1)        
        
        #zero_vec = self.xp.zeros(sub_vector_QK.shape, dtype=self.xp.float32)
        #(re1, im1) = F.fft((sub_vector_QK, zero_vec))
        #(re2, im2) = F.fft((sub_vector_V, zero_vec))
        #sub_vector = F.sum(F.ifft((re1*re2, im1*im2))[0], axis=1)
        return sub_vector


    def func_weighted_normalized_attn(self, subword_idx_k, subword_idx_v, subword_idx_q, ref_vector):
        batchsize = len(subword_idx_k)
        sp = (batchsize, self.embed_subword.W.shape[1])
        sub_vector = self.get_word_vec_attn(subword_idx_k, subword_idx_v, subword_idx_q)

        sub_vector_sq  = F.broadcast_to(F.sqrt(F.sum(sub_vector * sub_vector, axis=1, keepdims=True)), sp)
        sub_vector = sub_vector / sub_vector_sq  # 単位vector化

        ref_vector_sq0 = F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True))
        ref_vector_sq  = F.broadcast_to(ref_vector_sq0, sp)
        # ref_vector_sq  = F.broadcast_to(F.sqrt(F.sum(ref_vector * ref_vector, axis=1, keepdims=True)), sp)
        ref_vector = ref_vector / ref_vector_sq  # 単位vector化

        sq_loss = F.sum(F.squared_error(sub_vector, ref_vector)/self.embed_dim, axis=1, keepdims=True)
        return sq_loss




       

