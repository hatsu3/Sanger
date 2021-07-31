#!/bin/bash
export CONFIG_ROOT=$PROJ_ROOT/configs

model_name=${1:-"bert-base-uncased"}
model_config=${2:-"bert_base_sanger_2e-3.json"}
num_train_epochs=${3:-"2"}
learning_rate=${4:-"3e-5"}
batch_size=${5:-"12"}
output_dir=${6:-"$PROJ_ROOT/outputs/squad"}

python run_squad.py \
    --model_name_or_path sparse-$model_name \
    --config_name $CONFIG_ROOT/$model_config \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $output_dir/sparse-$(basename $model_name)/
