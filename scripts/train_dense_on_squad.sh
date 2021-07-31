#!/bin/bash
model_name=${1:-"bert-base-uncased"}
num_train_epochs=${2:-"2"}
learning_rate=${3:-"3e-5"}
batch_size=${4:-"12"}
output_dir=${5:-"$PROJ_ROOT/outputs/squad"}

python run_squad.py \
    --model_name_or_path $model_name \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $output_dir/$(basename $model_name)/
