#!/bin/bash
task_name=${1:-"mrpc"}
model_name=${2:-"bert-base-uncased"}
num_train_epochs=${3:-"3"}
learning_rate=${4:-"2e-5"}
batch_size=${5:-"32"}
output_dir=${6:-"$PROJ_ROOT/outputs/glue"}

python run_glue.py \
    --model_name_or_path $model_name \
    --task_name $task_name \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --output_dir $output_dir/$(basename $model_name)_$task_name/
