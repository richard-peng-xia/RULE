export CUDA_VISIBLE_DEVICES=1

cd /home/wenhao/Project/intern/kangyu/RULE/inference


python llava/eval/model_vqa_harvard.py \
    --model-path /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
    --question-file /home/wenhao/Project/intern/kangyu/RULE/data/annotations/inference/test/harvard_test_with-reference-top1.jsonl \
    --image-folder /home/wenhao/Datasets/med/Harvard/Test \
    --answers-file /home/wenhao/Project/intern/kangyu/RULE/checkpoints/output/inference_result/xxx.jsonl \
