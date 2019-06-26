# compact_reconstruction
- This repository is about *Subword-based Compact Reconstruction of Word Embeddings. Sasaki et al. NAACL2019*

## Table of contents
  - [Usage](#usage)
    - [Requirements](#requirements)
  - [Resources](#resources)


## Usage

### Requirements
- chainer
- numpy
- datetime

### How to train
```
$ src/train.py --gpu 0 \
--ref_vec_path crawl-300d-2M-subword.vec \
--freq_path freq_count.crawl-300d-2M-subword.vec\
--multi_hash two \
--maxlen 200 \
--codecs_path ngram_dic.crawl-subword.max30.min3.new \
--network_type 2 \
--subword_type 4 \
--limit_size 1000000 \
--bucket_size 100000 \
--result_dir ./result \
--hashed_idx \
--unique_false
```

### How to estimate (OOV) word vectors

## Resources
- Subword embeddings
  - SUM-F
  - SUM-H
  - KVQ-H
  - SUM-FH
  - KVQ-FH
  

