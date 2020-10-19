#!/bin/bash
source /home/haoranxu/Anaconda/python3/bin/activate clce

#$ -cwd
#$ -j y -o log/iter-norm-30-en
#$ -e erlog
#$ -m eas
#$ -M hxu64@jhu.edu
#$ -l ram_free=30g,gpu=1
#$ -pe smp 4
#$ -V

# rm -rf /export/b15/haoranxu/clce/outputs/iter-norm-tempp/

CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train allen_configs/enbert_IN.jsonnet -s /export/b15/haoranxu/clce/outputs/new2/enbert_IN-big/  --include-package src
