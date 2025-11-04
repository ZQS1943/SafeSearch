# SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents

<p align="left">
   <a href='https://arxiv.org/abs/2510.17017'>
    <img src='https://img.shields.io/badge/Arxiv-2510.17017-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
</p>

SafeSearch is a reinforcement learning framework that jointly optimizes safety and utility in LLM-based search agents.
Built upon [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [verl](https://github.com/volcengine/verl), it introduces safety training on red-teaming datasets and a dual-level reward that both rewards helpful, safe final responses and encourages safe intermediate query formulation.

This repository includes the full codebase for the paper, covering both the evaluation of system utility and safety, as well as the implementation of SafeSearch for aligning search agents across these dimensions.

![Overview of SafeSearch](assets/architecture.jpeg)

## Environment Setup
We recommend creating two separate environments (safesearch and retriever) to isolate dependencies.

### 1. SafeSearch
Install the environment.
```bash
conda create -y -n safesearch python=3.9
conda activate safesearch

# Torch (or skip and let vLLM pull a compatible version)
pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# vLLM (0.6.3 recommended; 0.5.4 / 0.4.2 / 0.3.1 also compatible)
pip install "vllm==0.6.3"

# FlashAttention 2 (optional; faster; may require Py3.10/CUDA toolchain)
pip install flash-attn --no-build-isolation || echo "flash-attn optional; continuing"

pip install -r requirements.txt
```

### 2. Local Retriever
(1) Download the indexing and corpus.
```bash
SAVE_DIR="./corpus"
mkdir -p "$SAVE_DIR"

# Clone both datasets
git clone https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus "$SAVE_DIR/wiki-18-corpus"
git clone https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index "$SAVE_DIR/wiki-18-e5-index"

# Decompress corpus
gzip -df "$SAVE_DIR/wiki-18-corpus/wiki-18.jsonl.gz" || true

# Concatenate index parts
cat "$SAVE_DIR/wiki-18-e5-index"/part_* > "$SAVE_DIR/wiki-18-e5-index/e5_Flat.index"
```

(2) Install the environment.
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

# Install the GPU version of faiss for efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

## Data Preparation
Our datasets are derived from publicly available sources and processed into the required format for SafeSearch.
```bash
cd data
```
### 1. Evaluation Data
(1) Safety Eval
We use three red-teaming datasets for safety evaluation:
```bash
git clone https://github.com/haizelabs/redteaming-resistance-benchmark.git
git clone https://github.com/alexandrasouly/strongreject.git
git lfs install
git clone https://huggingface.co/datasets/allenai/wildjailbreak
python data_processing.py --data_type eval_safety 
```

(2) Utility/QA Eval
We use three QA datasets — TriviaQA (test), HotpotQA (dev), and Bamboogle (test) — to evaluate model utility.
Download them from [FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) and organize them as follows:
```bash
SafeSearch
├── data
│   └── FlashRAG_datasets
│       ├── bamboogle
│       │   └── test.jsonl
│       ├── hotpotqa
│       │   └── dev.jsonl
│       └── triviaqa
│           └── test.jsonl

```
Then run: 
```bash
python data_processing.py --data_type eval_utility
```

### 1. Training Data
SafeSearch finetunes the LLM using a mixed dataset that combines safety and utility data.
(1) Safety Training
Red-teaming instructions are sampled from the training split of the WildTeaming dataset:
```bash
git lfs install
git clone https://huggingface.co/datasets/allenai/wildjailbreak # Skip if already cloned
python data_processing.py --data_type finetune_safety
```

(2) Utility Training
The utility training data contains the same QA pairs as [Search-R1](https://github.com/PeterGriffinJin/Search-R1), but with modified prompts:
```bash
python data_processing.py --data_type finetune_utility
```



## Quick Start
For either evaluation or finetuning, we need to start the local retriever first:
```bash
conda activate retriever
bash scripts/retriever_launch.sh gpu
```

### 1. Evaluation
To conduct safety and utility evaluation:
```bash
conda activate safesearch
export PYTHONPATH=.
bash scripts/eval.sh
```

### 2. Training
To run SafeSearch, use:
```bash
conda activate safesearch
export PYTHONPATH=.
export WANDB_API_KEY=your_wandb_key
bash scripts/safesearch_train.sh
```

## Note

SafeSearch uses AWS Bedrock to call gpt-oss-20b as the LLM judge (see `src/utils/llm.py`).
To enable this, you must also specify your Bedrock token:

```bash
export AWS_BEARER_TOKEN_BEDROCK=your_bedrock_token
```

Alternatively, you can implement your own client following the same interface as `_OPENAIClient`.


## Results
SafeSearch significantly reduces harmful response rates across red-teaming datasets, while maintaining helpful and safe outputs instead of unhelpful hard refusals. For more detailed analysis, please refer to the paper.
![Result](assets/results.png)

## Acknowledgment
The SafeSearch training code is adapted from [Search-R1](https://github.com/PeterGriffinJin/Search-R1), and the evaluation code is adapted from [Search-o1](https://github.com/RUC-NLPIR/Search-o1).

This work was primarily developed by [Qiusi Zhan](https://zqs1943.github.io/), with contributions from Angeline Budiman-Chan, Abdelrahman Zayed, Xingzhi Guo, Daniel Kang, and Joo-Kyung Kim.

## Citation
```bibtex
@article{zhan2025safesearch,
  title={SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents},
  author={Zhan, Qiusi and Budiman-Chan, Angeline and Zayed, Abdelrahman and Guo, Xingzhi and Kang, Daniel and Kim, Joo-Kyung},
  journal={arXiv preprint arXiv:2510.17017},
  year={2025}
}
```

## Contributing
See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
This library is licensed under [the CC-BY-NC-4.0 License](LICENSE).

