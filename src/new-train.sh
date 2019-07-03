#!/usr/bin/env bash

#$ -cwd
#$ -ac d=nvcr-cuda-9.0-cudnn7
. /fefs/opt/dgx/env_set/nvcr-cuda-9.0-cudnn7.sh
# source /fefs/opt/dgx/env_set/common_env_set.sh


export PYENV_ROOT=$HOME/.pyenv/
export PATH=$PYENV_ROOT/plugins/pyenv-virtualenv/shims:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell anaconda3-5.1.0/envs/compact-recon

echo "JOB ID: " $JOB_ID
date

echo "Start: train"
echo "/home/sasaki/sasaki_work/projects/compact_reconstruction/src/train.py \
    --gpu 0 \
    --ref_vec_path $REF_VEC \
    --freq_path $FREQ \
    --multi_hash $M_HASH \
    --maxlen $MAXLEN \
    --codecs_path $CODECS \
    --network_type $NET_TYPE \ 
    --subword_type $SUB_TYPE \
    --limit_size $LIMIT \
    --bucket_size $BUCKET \
    --result_dir /home/sasaki/sasaki_work/projects/compact_reconstruction/result\
    $TRAIN_FLAGS"

python /home/sasaki/sasaki_work/projects/compact_reconstruction/src/train.py \
    --gpu 0 \
    --ref_vec_path $REF_VEC \
    --freq_path $FREQ \
    --multi_hash $M_HASH \
    --maxlen $MAXLEN \
    --codecs_path $CODECS \
    --network_type $NET_TYPE \
    --subword_type $SUB_TYPE \
    --limit_size $LIMIT \
    --bucket_size $BUCKET \
    --result_dir /home/sasaki/sasaki_work/projects/compact_reconstruction/result\
    $TRAIN_FLAGS

exit


if [ $? -ne 0 ]; then
   echo "#FAIL 1"
   exit;
else
   :
fi
echo "End: train"

VAR=$(cat /home/sasaki/sasaki_work/subword_vector/ngram_sq_kvq/jobid-to-savepath/$JOB_ID)
# echo $VAR

echo "Start: save"
echo "python /home/sasaki/sasaki_work/subword_vector/ngram_sq_kvq/src/save_embedding.py \
    --raiden \
    --gpu 0 \
    --inference \
    --model_path ${VAR}model_epoch_300 \
    $SAVE_FLAGS"

python /home/sasaki/sasaki_work/subword_vector/ngram_sq_kvq/src/save_embedding.py \
    --raiden \
    --gpu 0 \
    --inference \
    --model_path ${VAR}model_epoch_300 \
    $SAVE_FLAGS

echo "End: save"

if [ `echo ${SAVE_FLAGS} | grep '\-\-test_word_file_path'|sed -e 's/ //g'` ] ; then
  echo ${VAR}embedding_epoch300/embedding_reconstruct_original.add_test_words.txt > /home/sasaki/sasaki_work/subword_vector/ngram_sq_kvq/eval_queue/$JOB_ID
else
  echo ${VAR}embedding_epoch300/embedding_reconstruct_original.txt > /home/sasaki/sasaki_work/subword_vector/ngram_sq_kvq/eval_queue/$JOB_ID
fi
date
