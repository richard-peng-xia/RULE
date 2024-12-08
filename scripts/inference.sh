#!/bin/bash

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=2
dataset='iuxray' # 'harvard' 'mimic'

python llava/eval/model_vqa_${dataset}.py \
    --model-base /path/to/llava-med-1.5_model_weight \
    --model-path /path/to/lora/weight \
    --question-file /path/to/question.jsonl \
    --image-folder /path/to/image_folder \
    --answers-file /path/to/output.jsonl/saving/location
