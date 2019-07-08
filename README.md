# Compact Reconstruction
- This repository is about *Subword-based Compact Reconstruction of Word Embeddings. Sasaki et al. NAACL2019*

## Table of contents
  - [Usage](#usage)
    - [Requirements](#requirements)
    - [How to train](#how-to-train)
    - [How to estimate (OOV) word vectors](#how-to-estimate-oov-word-vectors)    
  - [Preprocessing of setting files](#preprocessing-of-setting-files)
  - [Resources](#resources)


## Usage

### Requirements
- Python version >= 3.7
- chainer
- numpy
- datetime

### How to train

```
$ src/train.py \
--gpu 0 \
--ref_vec_path crawl-300d-2M-subword.vec \
--freq_path resources/freq_count.crawl-300d-2M-subword.vec \
--multi_hash two \
--maxlen 200 \
--codecs_path resources/ngram_dic.max30.min3 \
--network_type 2 \
--subword_type 4 \
--limit_size 1000000 \
--bucket_size 100000 \
--result_dir ./result \
--hashed_idx \
--unique_false
```
||net_type  |subword_type  |hashed_idx  |
|---|---|---|---|
|SUM-F  |2  |0  |✘  |
|SUM-H  |2  |0  |✓  |
|KVQ-H  |3  |0  |✓  |
|SUM-FH  |2  |4  |✓  |
|KVQ-FH  |3  |4  |✓  |

### How to estimate (OOV) word vectors

For estimating OOV word vectors:
```
$ python src/inference.py \
--gpu 0 \
--model_path \
result/sum/20190625_00_57_18/model_epoch_300 \
--codecs_path resources/ngram_dic.max30.min3 \
--oov_word_path resources/oov_words.txt
```

For reconstructing original word embeddings:
```
$ python src/save_embedding.py \
--gpu 0 \
--inference \
--model_path result/sum/20190625_00_57_18/model_epoch_300
```

## Preprocessing of setting files
- See [preprocessing page](https://github.com/losyer/compact_reconstruction/tree/master/src/preprocess)

## Resources
- See [resource page](https://github.com/losyer/compact_reconstruction/tree/master/resources)
  

