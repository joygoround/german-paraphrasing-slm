# Multi-Level German Text Simplification with Small Language Models

Source code for the thesis *"De-Paraphrasing with Small Language Models: Multi-Level German Text Simplification using LFM2.5"*.

All experiments use [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) and its predecessor [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) as backbone models, fine-tuned on the [German4All-Corpus](https://huggingface.co/datasets/tum-nlp/German4All-Corpus).

---

## Repository structure

```
repo/
├── config_train.py              # Hyperparameters and model configs for all training strategies
├── training_utils.py            # Shared utilities (checkpointing, memory, logging)
│
├── train_allcomb.py             # Strategy 1: All-Combination SFT
├── train_contrastive.py         # Strategy 2: Contrastive Learning (InfoNCE)
├── train_multitask.py           # Strategy 3: Multitask Learning (complexity classification head)
│
├── complexity_desc.txt          # Five-level complexity taxonomy embedded in every prompt
│
├── eval_generate.py             # Generate outputs for all models → outputs CSV
├── eval_reference.py            # SARI, BERTScore F1, chrF
├── eval_surface.py              # WSTF, LIX, FRE_DE, structural counters
├── eval_faithfulness.py         # Named Entity Retention Rate (spaCy)
├── eval_detect.py               # DETECT score (Simplicity, Meaning, Fluency)
├── eval_llm_judge_tgi.py        # LLM-as-a-Judge via local TGI server
│
├── eval_config.py               # Model paths and generation parameters
├── eval_utils.py                # Shared eval utilities (generation, readability)
├── eval_prompt.py               # All prompts (generation + judge)
│
└── requirements.txt
```

## Fine-tuned models

All fine-tuned checkpoints are available on HuggingFace under the `joygoround/` namespace (see Table 4.1 in the thesis). `eval_config.py` contains the full model path mapping.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_lg
```

For training on AMD GPUs, install the ROCm build of PyTorch separately:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

The DETECT metric requires a manual install from [https://github.com/ZurichNLP/DETECT](https://github.com/ZurichNLP/DETECT). Update `DETECT_PATH` in `eval_detect.py` after cloning.

---

## Training

Edit the active model entries in `config_train.py`, then run the desired strategy:

```bash
python train_allcomb.py
python train_contrastive.py
python train_multitask.py
```

All three scripts read hyperparameters from `config_train.py` and training utilities from `training_utils.py`.

---

## Evaluation

**Step 1 — Generate outputs** (produces a single CSV used by all downstream scripts):
```bash
python eval_generate.py
```

**Step 2 — Run metrics** (each takes the outputs CSV as argument):
```bash
python eval_reference.py   <outputs.csv>
python eval_surface.py     <outputs.csv>
python eval_faithfulness.py <outputs.csv>
python eval_detect.py      <outputs.csv>
```

**Step 3 — LLM-as-a-Judge** (requires a running TGI server and `TGI_BASE_URL` in `.env`):
```bash
python eval_llm_judge_tgi.py <outputs.csv>
# Resume an interrupted run:
python eval_llm_judge_tgi.py <outputs.csv> --resume <checkpoint.csv>
```