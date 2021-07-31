#!/bin/bash
model_name=${1:-"bert-base-uncased"}
num_train_epochs=${2:-"20"}
learning_rate=${3:-"5e-5"}
batch_size=${4:-"3"}
output_dir=${5:-"$PROJ_ROOT/outputs/cloth"}

python run_cloth.py \
    --model_name_or_path $model_name \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --train_batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --output_dir $output_dir/$(basename $model_name)/
