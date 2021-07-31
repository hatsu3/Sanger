#!/bin/bash
model_name=${1:-"bert-base-uncased"}
num_train_epochs=${2:-"20"}
output_dir=${3:-"$PROJ_ROOT/outputs/cloth"}

python run_cloth.py \
    --model_name_or_path $model_name \
    --do_eval \
    --max_seq_length 512 \
    --num_train_epochs $num_train_epochs \
    --eval_checkpoint $output_dir/$(basename $model_name)/checkpoint_${num_train_epochs} \
    --output_dir $output_dir/$(basename $model_name)/
