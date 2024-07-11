import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
import re
from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()
import debugpy
# rank = int(os.getenv('RANK', '0'))
# port = 5678 + rank  # 基础端口 + 进程ID

# debugpy.listen(port)
# print(f"Process {rank} waiting for debugger to attach on port {port}...")
# debugpy.wait_for_client()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["question"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        
        if "reference_report" not in line:
            # continue
            
            # cur_prompt = qs
            gt_answer=line['answer']
            suffix='Please answer the question based on the image and report and choose from the following two options: [yes, no].'
            cur_prompt = qs+' '+suffix
            qs=cur_prompt
        
        else:
            reference_report=line["reference_report"]
            gt_answer=line["answer"]
            # print(reference_report)
            if not isinstance(reference_report, list):
                topk=1
                reference_report=[reference_report]
                # formatted_reference_report=reference_report[0]
                cleaned_report = reference_report[0].replace('\n', ' ')

                # # 去除"Impression:"部分
                # cleaned_report = re.sub(r'[Ii]mpression:.*?(?=(Findings:|findings:|$))', '', cleaned_report, flags=re.DOTALL)

                # # 去除"Findings:"部分
                # cleaned_report = re.sub(r'\b[Ff]indings:\s*', '', cleaned_report)

                # 将其余文本连接起来
                cleaned_report = ' '.join(cleaned_report.split())

                # 只保留前三句
                # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cleaned_report)
                # cleaned_report = ' '.join(sentences[:2])

                formatted_reference_report = f"{cleaned_report} "
            else:
                topk=len(reference_report)
                formatted_reference_report=""
                for i in range(topk):
                    # formatted_reference_report += f"{i + 1}. {reference_report[i]} "
                    cleaned_report = reference_report[i].replace('\n', ' ')
                    # cleaned_report = reference_report[i].replace('___', 'xxx')
                    
                    # cleaned_report = re.sub(r'\b[Ii]mpression:\s*', '', cleaned_report)
                    # cleaned_report = re.sub(r'\b[Ff]indings:\s*', '', cleaned_report)


                    # cleaned_report = re.sub(r'[Ii]mpression:.*?(?=(Findings:|findings:|$))', '', cleaned_report, flags=re.DOTALL)

                    # # 使用正则表达式去除"Findings:"或"findings:"标记
                    # cleaned_report = re.sub(r'\b[Ff]indings:\s*', '', cleaned_report)

                    # 将其余文本连接起来
                    cleaned_report = ' '.join(cleaned_report.split())

                    formatted_reference_report += f"{i + 1}. {cleaned_report} "


                # print(formatted_reference_report)
            # cur_prompt = qs
            # suffix='Please answer the question based on the image and report and choose from the following two options: [yes, no]. Please directly answer starting with "Yes" or "No". The answer should be limited to two sentences.'

            # appendix_1=f"You are provided with a chest X-ray image, a image-related question and {topk} reference report(s): "
            # appendix_2="It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, but should only be used for reference and comparison. Question: "
            # cur_prompt = appendix_1 + formatted_reference_report +"\n"+ appendix_2 +"\n"+qs

            appendix_1=f"You are provided with a chest X-ray image, a image-related question: \n"
            appendix_2=f"You are also provided with {topk} reference report(s).Please answer the question based on the image and report and answer the question based on the image and report and choose from the following two options: [yes, no]. It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, but should only be used for reference. \nReference reports:"
            cur_prompt = appendix_1 +qs+ "\n"+appendix_2 +"\n"+formatted_reference_report
            # If the reference reports are too long and get cut off, do not refill the remaining sentence; directly answer the question.

            # Please directly answer starting with 'Yes' or 'No'. The answer should be limited to one sentence.
            # print(cur_prompt)
            qs=cur_prompt
        # cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


        # 检查 input_ids 的长度
        # input_length = input_ids.shape[1]
        # print(f"Input length: {input_length}")

        # # 获取模型的最大输入长度
        # max_input_length = model.config.max_position_embeddings
        # print(f"Max input length: {max_input_length}")

        # if input_length > max_input_length:
        #     print(f"Input is truncated. Original length: {input_length}, Max length: {max_input_length}")


        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
            # output = model.generate(
            #     input_ids,
            #     images=image_tensor.unsqueeze(0).half().cuda(),
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     max_new_tokens=1024,
            #     use_cache=True,
            #     output_scores=True,
            #     return_dict_in_generate=True)

        # output_ids = output.sequences
        # scores = output.scores

        # # 计算每个token的概率
        # probabilities = []
        # for i, score in enumerate(scores):
        #     softmax_scores = torch.nn.functional.softmax(score, dim=-1)
        #     probabilities.append(softmax_scores)

        # 输出的token序列
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "answer": outputs,
                                   "gt_answer":gt_answer,
                                   "image":image_file,
                                   "image_id":image_file,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
