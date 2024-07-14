#!/bin/bash

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=2

# Check the first argument to choose the dataset
if [ "$1" = "iuxray" ]; then
    python llava/eval/model_vqa_iuxray.py \
        --model-base /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --model-path /home/wenhao/Project/intern/kangyu/RULE/checkpoints/output/iuxray_lora-qrefVqa \
        --question-file /home/wenhao/Project/intern/kangyu/RULE/data/annotations/inference/test/iuxray_test_with-reference-top1.jsonl \
        --image-folder /home/wenhao/Datasets/med/rad/iu_xray/images \
        --answers-file /home/wenhao/Project/intern/kangyu/RULE/checkpoints/output/inference_result/xxx.jsonl
elif [ "$1" = "harvard" ]; then
    python llava/eval/model_vqa_harvard.py \
        --model-base /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --model-path /home/wenhao/Project/intern/kangyu/RULE/checkpoints/output/iuxray_lora-qrefVqa \
        --question-file /home/wenhao/Project/intern/kangyu/RULE/data/annotations/inference/test/harvard_test_with-reference-top1.jsonl \
        --image-folder /home/wenhao/Datasets/med/Harvard/Test \
        --answers-file /home/wenhao/Project/intern/kangyu/RULE/checkpoints/output/inference_result/xxx.jsonl
else
    python llava/eval/model_vqa_mimic.py \
        --model-base /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --model-path /home/wenhao/Project/intern/kangyu/RULE/checkpoints/output/mimic_lora-qrefVqa \
        --question-file /home/wenhao/Project/intern/kangyu/RULE/data/annotations/inference/test/mimic_test_with-reference-top1.jsonl \
        --image-folder /home/wenhao/Datasets/med/rad/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files \
        --answers-file /home/wenhao/Project/intern/kangyu/RULE/checkpoints/output/inference_result/xxx.jsonl
fi