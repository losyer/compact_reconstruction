# Preprocess

## Word frequency dictionary
In our framework, the system use a word frequency dictionary for the reconstruction training.

If you don't have a word frequency dictionary, but only have pretrained word embeddings that is available online (e.g. GloVe.600B vectors),
you can obtain a word frequency dictionary by using `complement_freq_dic.py`.


1. Prepare source word frequency dictionary.
2. Use `complement_freq_dic.py` like this:

```
$ python complement_freq_dic.py \
--org_freq_path  [source word frequency dictionary] \
--target_path [target vector (word2vec format)] \
> [output dictionary file]
```


## Character N-gram dictionary

### How to make character N-gram dictionary
```
$ python make_ngram_dic.py \
--ref_vec_path [reference_vector_path (word2vec format)]  \
--output tmp_out \
--n_max 30 \
--n_min 3

$ sort -k 2,2 -n -r tmp_out > outfile-min3max30.sorted
```
