#!/bin/bash

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=2

# Check the first argument to choose the dataset
if [ "$1" = "iuxray" ]; then
    python llava/eval/model_vqa_iuxray.py \
        --model-base LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --model-path checkpoints/output/iuxray_lora-qrefVqa \
        --question-file QUESTION_PATH \
        --image-folder iu_xray/images \
        --answers-file OUTPUT_PATH
elif [ "$1" = "harvard" ]; then
    python llava/eval/model_vqa_harvard.py \
        --model-base LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --model-path checkpoints/output/iuxray_lora-qrefVqa \
        --question-file QUESTION_PATH \
        --image-folder Harvard/images \
        --answers-file OUTPUT_PATH
else
    python llava/eval/model_vqa_mimic.py \
        --model-base LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --model-path checkpoints/output/mimic_lora-qrefVqa \
        --question-file QUESTION_PATH \
        --image-folder mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files \
        --answers-file OUTPUT_PATH
fi