#!/bin/bash
task_name=${1:-"mrpc"}
model_name=${2:-"bert-base-uncased"}
output_dir=${6:-"$PROJ_ROOT/outputs/glue"}

python run_glue.py \
    --model_name_or_path $model_name \
    --task_name $task_name \
    --do_eval \
    --max_seq_length 128 \
    --eval_checkpoint $output_dir/$(basename $model_name)_$task_name/pytorch_model.bin \
    --output_dir $output_dir/$(basename $model_name)_$task_name/
