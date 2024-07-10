# RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models

*The recent emergence of Medical Large Vision Language Models (Med-LVLMs) has enhanced medical diagnosis. However, current Med-LVLMs frequently encounter factual issues, often generating responses that do not align with established medical facts. Retrieval-Augmented Generation (RAG), which utilizes external knowledge, can improve the factual accuracy of these models but introduces two major challenges. First, limited retrieved contexts might not cover all necessary information, while excessive retrieval can introduce irrelevant and inaccurate references, interfering with the model's generation. Second, in cases where the model originally responds correctly, applying RAG can lead to an over-reliance on retrieved contexts, resulting in incorrect answers. To address these issues, we propose RULE, which consists of two components. First, we introduce a provably effective strategy for controlling factuality risk through the calibrated selection of the number of retrieved contexts. Second, based on samples where over-reliance on retrieved contexts led to errors, we curate a preference dataset to fine-tune the model, balancing its dependence on inherent knowledge and retrieved contexts for generation. We demonstrate the effectiveness of RULE on three medical VQA datasets, achieving an average improvement of 20.8% in factual accuracy.*

<div align=left>
<img src=asset/logo.png width=90% />
</div>

**RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models** [[Paper](https://arxiv.org/abs/2407.05131)] <br>

## Schedule

+ [ ] Release the VQA data.
+ [ ] Release the training code.

## Installation


## Preparation
Download the data annotations from https://huggingface.co/datasets/zky11235/annotations

Download the model checkpoints after dpo training from https://huggingface.co/zky11235/dpo_checkpoints
## Training


## Inference


## Citation

```bibtex
@article{xia2024rule,
  title={RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models},
  author={Xia, Peng and Zhu, Kangyu and Li, Haoran and Zhu, Hongtu and Li, Yun and Li, Gang and Zhang, Linjun and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2407.05131},
  year={2024}
}
```

## Acknowledgement
We use code from [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), [POVID](https://github.com/YiyangZhou/POVID), [CARES](https://github.com/richard-peng-xia/CARES). We thank the authors for releasing their code.