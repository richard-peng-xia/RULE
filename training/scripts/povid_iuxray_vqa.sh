export CUDA_VISIBLE_DEVICES=2
# python llava/eval/model_vqa_iuxray.py \
#     --model-name /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava_med_model \
#     --question-file /home/wenhao/Datasets/med/rad/iu_xray/iu_xray_eval_qa.jsonl \
#     --image-folder /home/wenhao/Datasets/med/rad/iu_xray/images \
#     --answers-file /home/wenhao/Project/intern/xiapeng/med-dpo/outputs/answer-file.jsonl \
#     --conv-mode simple_legacy

# with reference report
# python llava/eval/model_vqa_iuxray.py \
#     --model-name /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava_med_model \
#     --question-file /home/wenhao/Project/intern/xiapeng/med-dpo/annotation/iu_xray_eval_qa_reference_1_all_finetune.jsonl \
#     --image-folder /home/wenhao/Datasets/med/rad/iu_xray/images \
#     --answers-file /home/wenhao/Project/intern/xiapeng/med-dpo/outputs/answer-file_reference_1_all_finetune.jsonl \
#     --conv-mode simple_legacy

# with 1 reference report
# python llava/eval/model_vqa_iuxray.py \
#     --model-name /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava_med_model \
#     --question-file /home/wenhao/Project/intern/xiapeng/med-dpo/annotation/iu_xray_eval_qa_reference_resnet_epoch360_top1.jsonl \
#     --image-folder /home/wenhao/Datasets/med/rad/iu_xray/images \
#     --answers-file /home/wenhao/Project/intern/xiapeng/med-dpo/outputs/answer-file_reference_resnet_epoch360_top1.jsonl \
#     --conv-mode simple_legacy

# with 1 reference report test set
# --model-name /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava_med_model \
# --model-name /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
# python llava/eval/model_llava_med_vqa.py \
python llava/eval/povid_model_vqa_iuxray.py \
    --model-base /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
    --model-path /home/wenhao/Project/intern/xiapeng/med-dpo/downloaded_files/POVID_stage_one_LoRa \
    --question-file /home/wenhao/Project/intern/xiapeng/med-dpo/annotation/iu_xray_eval_qa_reference_resnet_testSet_epoch360_top1.jsonl \
    --image-folder /home/wenhao/Datasets/med/rad/iu_xray/images \
    --answers-file /home/wenhao/Project/intern/xiapeng/med-dpo/outputs/answer-file_reference_resnet_testSet_epoch360_top1_lora_vqa.jsonl \
    # --conv-mode simple_legacy \
    # --lora_weight /home/wenhao/Project/intern/xiapeng/med-dpo/downloaded_files/POVID_stage_one_LoRa


# with 10 reference report
# python llava/eval/model_vqa_iuxray.py \
#     --model-name /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava_med_model \
#     --question-file /home/wenhao/Project/intern/xiapeng/med-dpo/annotation/iu_xray_eval_qa_reference_top10_all_finetune.jsonl \
#     --image-folder /home/wenhao/Datasets/med/rad/iu_xray/images \
#     --answers-file /home/wenhao/Project/intern/xiapeng/med-dpo/outputs/answer-file_reference_top10_all_finetune.jsonl \
#     --conv-mode simple_legacy

# python llava/eval/model_vqa.py     --model-name checkpoint/llava_med_model     --question-file data/iu_xray/iuxray_eval_qa.jsonl    --image-folder  /home/wenhao/Datasets/med/rad/iu_xray/images/     --answers-file data/iu_xray/answer-file.jsonl