#!/bin/bash
model_name=${1:-"bert-base-uncased"}
output_dir=${5:-"$PROJ_ROOT/outputs/squad"}

python run_squad.py \
    --model_name_or_path $model_name \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --eval_checkpoint $output_dir/$(basename $model_name)/pytorch_model.bin \
    --output_dir $output_dir/$(basename $model_name)/
