
export CUDA_VISIBLE_DEVICES='0,1,2,3'

deepspeed --master_port 29501 llava/train/train_dpo.py \
    --model_name_or_path /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path /home/wenhao/Project/intern/xiapeng/med-dpo/outputs/iuxray/dpo/iuxray_dpo_data_qref_vqa.json \
    --image_folder /home/wenhao/Datasets/med/rad/iu_xray/images  \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoint/output/iuxray_dpo_data_qref_vqa \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
