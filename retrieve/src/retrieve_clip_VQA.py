import torch
from PIL import Image
import open_clip
import argparse
import os
import json
import sys
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


from training.data import HarvardDataset,HarvardVQADataset
import debugpy
def retrieve_topk_per_image(logits, val_k_list,retrieve_threshold=''):
    
    if not retrieve_threshold:
        pred_per_val_image = []
        for i, k in enumerate(val_k_list):
            if k == 0:
                pred_per_val_image.append(torch.tensor([-1]))
            elif k == 1:
                pred_per_val_image.append(
                    logits["image_to_text"][i].argmax(dim=0, keepdim=True)
                )
            else:
                _, topk_indices = logits["image_to_text"][i].topk(k, dim=0)
                pred_per_val_image.append(topk_indices)
        return pred_per_val_image
    else:
        retrieve_threshold=float(retrieve_threshold)
        pred_per_val_image = []
        for i, k in enumerate(val_k_list):
            if k == 0:
                pred_per_val_image.append(torch.tensor([-1]))
            else:
                logit_values = logits["image_to_text"][i]
                top1_logit = logit_values.max()
                sorted_logits, sorted_indices = torch.sort(logit_values, descending=True)
                
                # Calculate the ratio of the top1 logit to the other logits
                ratios = top1_logit / sorted_logits
                
                # Select indices where the ratio is beyond the retrieve_threshold
                selected_indices = sorted_indices[ratios < retrieve_threshold]
                
                # If k is more than the chosen topn, just choose topn
                if selected_indices.size(0) > k:
                    selected_indices = selected_indices[:k]
                    
                pred_per_val_image.append(selected_indices)
        return pred_per_val_image


def get_logits(image_features, text_features, logit_scale):
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    return logits


def clean_data_info(data_info):
    cleaned_data_info = {}
    for key, value in data_info.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                cleaned_data_info[key] = value.item()
        elif isinstance(value, list) and len(value) == 1:
            cleaned_data_info[key] = value[0]
        else:
            cleaned_data_info[key] = value
    return cleaned_data_info


def split_and_clean_data_infos(batch_data_infos):
    cleaned_data_infos = []
    num_items = len(next(iter(batch_data_infos.values())))
    # for i in range(num_items):
    #     print(batch_data_infos)
    #     single_data_info = {key: value[i] for key, value in batch_data_infos.items()}
    for i in range(num_items):
        single_data_info = {}
        for key, value in batch_data_infos.items():
            if isinstance(value, torch.Tensor):
                single_data_info[key] = value[i]
            elif isinstance(value, list) and len(value) == num_items:
                single_data_info[key] = value[i]
            else:
                single_data_info[key] = value
        cleaned_data_infos.append(clean_data_info(single_data_info))
        # cleaned_data_info = clean_data_info(single_data_info)
        # cleaned_data_infos.append(cleaned_data_info)
    return cleaned_data_infos


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name_or_path,
        pretrained=args.checkpoint_path,
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name_or_path)

    train_dataset = HarvardDataset(
        args.img_root,
        args.train_json,
        preprocess,
        tokenizer,
        load_include_path=True,
    )
    eval_dataset = HarvardVQADataset(
        args.img_root,
        args.eval_jsonl,
        preprocess,
        tokenizer,
        # test=(args.eval_type=='test'),
        fixed_K=args.fixed_k,
        
        
        
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    eval_dataloader.num_samples = len(eval_dataset)
    eval_dataloader.num_batches = len(eval_dataloader)

    val_all_image_features, val_all_text_features, val_k_list = [], [], []
    train_all_image_features, train_all_text_features = [], []
    train_all_image_full_paths, val_all_image_full_paths = [], []
    data_infos_list = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(train_dataloader, desc="Extracting traininig features"):
            images, texts, image_full_paths = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            train_all_image_full_paths.extend(image_full_paths)

            model_out = model(images, texts)
            image_features, text_features, logit_scale = model_out
            train_all_image_features.append(image_features.cpu())
            train_all_text_features.append(text_features.cpu())
            logit_scale = logit_scale.mean()

        for batch in tqdm(eval_dataloader, desc="Extracting validation features"):
            images, texts, image_full_paths, retrieval_ks, data_infos = batch
            cleaned_data_infos = split_and_clean_data_infos(data_infos)
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            val_all_image_full_paths.extend(image_full_paths)
            model_out = model(images, texts)
            image_features, text_features, logit_scale = model_out
            val_all_image_features.append(image_features.cpu())
            val_all_text_features.append(text_features.cpu())
            logit_scale = logit_scale.mean()
            data_infos_list.extend(cleaned_data_infos)
            retrival_ks_list = [int(t) for t in retrieval_ks]
            
            val_k_list.extend(retrival_ks_list)
    print(f'val image feature length: {len(torch.cat(val_all_image_features))}')
    print(f'train text feature length: {len(torch.cat(train_all_text_features))}')
    logits = get_logits(
        image_features=torch.cat(val_all_image_features),
        text_features=torch.cat(train_all_text_features),
        logit_scale=logit_scale.cpu(),
    )
    
    if args.fixed_k:
        val_k_list = [int(args.fixed_k)]*len(val_k_list)
        print(f'fixed k: {args.fixed_k}')
    print(f'clip threshold: {args.clip_threshold}')
    pred_per_val_image = retrieve_topk_per_image(logits, val_k_list,retrieve_threshold=args.clip_threshold)
    true_val_k_list=[len(indices) for indices in pred_per_val_image]

    output_jsonl = args.output_path
    print(f'length of pred_per_val_image: {len(pred_per_val_image)}')
    print(f'length of data_infos_list: {len(data_infos_list)}')
    print(f'length of val_k_list: {len(val_k_list)}')
    with open(output_jsonl, "w") as f:
        for topk_indices, data_info, k in zip(
            pred_per_val_image, data_infos_list, true_val_k_list
        ):
            reference_reports = []
            
            for idx in topk_indices:
                if idx.item() == -1:
                    
                    break
                reference_reports.append(
                    train_dataloader.dataset.image_report_pairs[idx.item()][1]
                )
            
            data_info["reference_reports"] = reference_reports
            data_info["retrieve_k"] = k
            f.write(json.dumps(data_info))
            f.write("\n")
        print(f"Finished writing referenced reports to {output_jsonl}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve Top-K Predictions for Images"
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="",
        help="Path to image root directory",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="",
        help="Path to training JSON annotation file",
    )
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default="",
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="hf-hub:thaottn/OpenCLIP-resnet50-CC12M",
        help="Model name or path",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config_type", type=str, default="tokenProb", help="Configuration type"
    )
    parser.add_argument(
        "--output_path", type=str, default="", help="Path to output JSONL file"
    )
    parser.add_argument(
        "--clip_threshold",type=str,default='',help='clip threshold for retrieval'
    )
    parser.add_argument(
        "--fixed_k",type=str,default='',help='fixed k for retrieval'
    )
    parser.add_argument(
        "--eval_type",type=str,default='',help='eval type'
    )
    

    args = parser.parse_args()
    main(args)
