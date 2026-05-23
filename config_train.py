# config_train.py
# Shared training configuration for train_allcomb.py, train_contrastive.py, train_multitask.py
# Hyperparameters correspond to Table 4.2 in the thesis.

import torch

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


def get_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    gpu_name = torch.cuda.get_device_name(0)
    if 'radeon' in gpu_name.lower() or 'amd' in gpu_name.lower():
        return torch.bfloat16
    try:
        major = torch.cuda.get_device_properties(0).major
        return torch.bfloat16 if major >= 8 else torch.float16
    except Exception:
        return torch.float16

DTYPE = get_dtype()


# ─────────────────────────────────────────────────────────────
# Model configurations
# Comment/uncomment entries to select which backbone to train.
# ─────────────────────────────────────────────────────────────

MODEL_CONFIGS_ALLCOMB = [
    {
        "name": "LFM2.5-1.2B-Instruct",
        "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
        "template_type": "lfm2.5",
        "device_map": "auto",
        "torch_dtype": DTYPE,
        "output_prefix": "lfm2.5_allcomb",
    },
    # {
    #     "name": "LFM2-1.2B",
    #     "model_id": "LiquidAI/LFM2-1.2B",
    #     "template_type": "lfm2",
    #     "device_map": "auto",
    #     "torch_dtype": DTYPE,
    #     "output_prefix": "lfm2_allcomb",
    # },
]

MODEL_CONFIGS_CONTRASTIVE = [
    {
        "name": "LFM2.5-1.2B-Instruct",
        "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
        "template_type": "lfm2.5",
        "device_map": "auto",
        "torch_dtype": DTYPE,
        "output_prefix": "lfm2.5_contrastive",
    },
    # {
    #     "name": "LFM2-1.2B",
    #     "model_id": "LiquidAI/LFM2-1.2B",
    #     "template_type": "lfm2",
    #     "device_map": "auto",
    #     "torch_dtype": DTYPE,
    #     "output_prefix": "lfm2_contrastive",
    # },
]

MODEL_CONFIGS_MULTITASK = [
    {
        "name": "LFM2.5-1.2B-Instruct",
        "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
        "template_type": "lfm2.5",
        "device_map": "auto",
        "torch_dtype": DTYPE,
        "output_prefix": "lfm2.5_multitask",
    },
    # {
    #     "name": "LFM2-1.2B",
    #     "model_id": "LiquidAI/LFM2-1.2B",
    #     "template_type": "lfm2",
    #     "device_map": "auto",
    #     "torch_dtype": DTYPE,
    #     "output_prefix": "lfm2_multitask",
    # },
]


# ─────────────────────────────────────────────────────────────
# Chat templates
# Only the LFM2 and LFM2.5 ChatML templates are used in this
# study. The assistant prefix marks the loss masking boundary.
# ─────────────────────────────────────────────────────────────

CHAT_TEMPLATES = {
    "lfm2.5": {
        "prompt_template": (
            "<|startoftext|><|im_start|>system\n"
            "You are a helpful assistant trained by Liquid AI for German paraphrasing."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{target_text}"
            "<|im_end|>"
        ),
        "assistant_prefix": "<|im_start|>assistant\n",
    },
    "lfm2": {
        "prompt_template": (
            "<|startoftext|>"
            "<|im_start|>system\n"
            "{prompt}"
            "<|im_end|>"
            "<|im_start|>assistant\n"
            "{target_text}"
            "<|im_end|>"
        ),
        "assistant_prefix": "<|im_start|>assistant\n",
    },
}


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

DATASET_ID = "tum-nlp/German4All-Corpus"
DATASET_DIR = "main"
COMPLEXITY_LEVELS = [1, 2, 3, 4, 5]
TARGET_COL_PREFIX = "cl_"
SRC_COL = "text"


# ─────────────────────────────────────────────────────────────
# Training hyperparameters
# Three separate dicts, one per strategy, matching Table 4.2.
# Shared values: 1 epoch, bfloat16, gradient checkpointing,
# effective batch size 16, max context 1024 tokens.
# ─────────────────────────────────────────────────────────────

_SHARED = {
    "num_train_epochs": 1,
    "bf16": DTYPE == torch.bfloat16,
    "fp16": False,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "weight_decay": 0.01,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,
    "logging_steps": 50,
    "logging_first_step": True,
    "report_to": "tensorboard",
    "remove_unused_columns": False,
    "ddp_find_unused_parameters": False,
}

# AllComb SFT — AdamW 8-bit, lr 5e-6, grad clip 1.0, batch 2 × 8 = 16
TRAINING_ARGS_ALLCOMB = {
    **_SHARED,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-6,
    "optim": "adamw_8bit" if BITSANDBYTES_AVAILABLE else "adamw_torch",
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 2,
}

# Contrastive — AdamW 8-bit, lr 5e-6, grad clip 1.0, batch 1 × 16 = 16, no eval
TRAINING_ARGS_CONTRASTIVE = {
    **_SHARED,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-6,
    "optim": "adamw_8bit" if BITSANDBYTES_AVAILABLE else "adamw_torch",
    "max_grad_norm": 1.0,
    "eval_strategy": "no",
    "dataloader_num_workers": 0,   # required for negative_input_ids passthrough
    "dataloader_pin_memory": True,
}

# Multitask — standard AdamW (32-bit), lr 2e-6, grad clip 0.3, batch 2 × 8 = 16
TRAINING_ARGS_MULTITASK = {
    **_SHARED,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-6,
    "optim": "adamw_torch",        # standard 32-bit for numerical stability
    "max_grad_norm": 0.3,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "eval_accumulation_steps": 1,
    "eval_do_concat_batches": False,
    "save_safetensors": False,
    "logging_nan_inf_filter": True,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 2,
}

# Auxiliary classification loss weight λ (see Section 4.3.3)
LAMBDA_COMPLEXITY = 0.005


# ─────────────────────────────────────────────────────────────
# Paths  —  adjust BASE_OUTPUT_DIR for your environment
# ─────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR = ""

RESULTS_DIR = f"{BASE_OUTPUT_DIR}/results"
RUNS_DIR = f"{BASE_OUTPUT_DIR}/runs"
CACHE_DIR = f"{BASE_OUTPUT_DIR}/.cache/datasets"

COMPLEXITY_DESC_PATH = ""
COMPLETION_FLAG_DIR = BASE_OUTPUT_DIR