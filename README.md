# [EMNLP'24] RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models

*We tackle the challenge of improving factual accuracy in Medical Large Vision Language Models (Med-LVLMs) using our novel approach, RULE. Despite their promise, Med-LVLMs often generate responses misaligned with established medical facts. RULE addresses this with two key strategies: 1) Calibrated selection of retrieved contexts to control factuality risk. 2) Fine-tuning models using a preference dataset to balance reliance on inherent knowledge and retrieved contexts. Our method achieves a 20.8% improvement in factual accuracy across three medical VQA datasets.*

**RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models** [[Paper](https://arxiv.org/abs/2407.05131)] <br>
<div align=left>
<img src=asset/logo.png width=90% />
</div>

## 🌟 Requirements
1. Clone this repository and navigate to RULE folder
```bash
git clone https://github.com/richard-peng-xia/RULE.git
cd RULE
```

2. Install Package: Create conda environment

```Shell
conda create -n RULE python=3.10 -y
conda activate RULE
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install trl
```

3. Download the required model checkpoints [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) from huggingface.

4. For all the medical datasets, you need firstly apply for the right of access and then download the dataset.

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) (Thanks to [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) for sharing the file)
- [Harvard-FairVLMed](https://ophai.hms.harvard.edu/datasets/harvard-fairvlmed10k/)

## 📖 Data Description
We provide a corresponding json or jsonl file for each dataset, including the image path, question, answer, and original report.

- Training: The data used to train the retriever and fine-tune the Med-LVLM are located in `data/training/retriever` and `data/training/alignment` respectively. 

- Test: All the test data for Med-LVLMs is placed under `data/test`. 

## 🚀 Training

- The training code of Direct Preference Optimization is at `llava/train/train_dpo.py`. 
- The relevant script can be found at `scripts/run_dpo.sh`


## 🛠️ Inference

- For test dataset inference, you need to specify the following arguments.
```python
python llava/eval/model_vqa_{dataset}.py \
    --model-base 'path/to/llava-med-1.5' \
    --model-path 'path/to/lora_weights' \
    --question-file 'path/to/question_file.json' \
    --image-folder 'path/to/test_images' \
    --answers-file 'path/to/output_file.json'
```
- The written script is at `scripts/inference.sh`. Before that, you need to set the correct path of data and checkpoints in your script.

## 📚 Citation

```bibtex
@article{xia2024rule,
  title={RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models},
  author={Xia, Peng and Zhu, Kangyu and Li, Haoran and Zhu, Hongtu and Li, Yun and Li, Gang and Zhang, Linjun and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2407.05131},
  year={2024}
}
```

## 🙏 Acknowledgement
We use code from [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), [POVID](https://github.com/YiyangZhou/POVID), [CARES](https://github.com/richard-peng-xia/CARES). We thank the authors for releasing their code.
