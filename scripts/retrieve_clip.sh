cd ./train/open_clip/src || exit
CUDA_VISIBLE_DEVICES=1 python ./retrieve_clip_VQA.py \
    --img_root /path/to/image_folder \
    --train_json /path/to/the/source/json/containing/gt_report \
    --eval_json /path/to/the/to/be/retrieved/json_file \
    --model_name_or_path hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --checkpoint_path /path/to/the/fine-tuned/clip_model \
    --output_path /path/to/the/output/json_file \
    --fixed_k /number/of/report/to/retrieve \



