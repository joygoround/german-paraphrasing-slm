# -*- coding: utf-8 -*-
import os
import random
import sys
import gc
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from config_train import (
    MODEL_CONFIGS_ALLCOMB as SINGLE_MODEL_CONFIGS,
    CHAT_TEMPLATES,
    DATASET_ID,
    DATASET_DIR,
    SRC_COL,
    TARGET_COL_PREFIX,
    COMPLEXITY_LEVELS,
    TRAINING_ARGS_ALLCOMB as TRAINING_ARGS,
    BITSANDBYTES_AVAILABLE,
    COMPLEXITY_DESC_PATH,
    COMPLETION_FLAG_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    CACHE_DIR,
)

from training_utils import (
    cleanup_memory,
    print_gpu_memory,
    cleanup_incomplete_checkpoints,
    find_latest_checkpoint,
    create_completion_flag,
    check_completion_flag,
    load_complexity_description,
    print_training_summary,
)

IGNORE_INDEX = -100


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        attention_mask = (input_ids_padded != self.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
        }


def create_all_combinations_data(dataset_split, full_description):
    all_data = []
    all_sources = ["ORIGINAL"] + [f"Level {l}" for l in COMPLEXITY_LEVELS]

    for ex in dataset_split:
        text_map = {"ORIGINAL": ex[SRC_COL]}
        for l in COMPLEXITY_LEVELS:
            col = f"{TARGET_COL_PREFIX}{l}"
            if col in ex and ex[col]:
                text_map[f"Level {l}"] = ex[col]

        for tgt_level in COMPLEXITY_LEVELS:
            tgt_key = f"Level {tgt_level}"
            if tgt_key not in text_map:
                continue

            for src_key in all_sources:
                if src_key not in text_map or src_key == tgt_key:
                    continue

                prompt = (
                    f"{full_description}\n\n"
                    f"### AUFGABE:\n"
                    f"Paraphrasiere den folgenden Quelltext von {src_key} auf {tgt_key}.\n\n"
                    f"### QUELLTEXT:\n{text_map[src_key]}\n\n"
                    f"### ZIEL-LEVEL:\n{tgt_key}\n\n"
                    f"### AUSGABEFORMAT:\n"
                    f"Gib NUR den paraphrasierten deutschen Text aus."
                )

                all_data.append({
                    "prompt": prompt,
                    "target_text": text_map[tgt_key],
                })

    print(f"Generated {len(all_data)} training samples")
    return all_data


def apply_template(data, template):
    return [
        {"text": template.format(prompt=d["prompt"], target_text=d["target_text"])}
        for d in data
    ]


def preprocess_dataset(data, tokenizer, assistant_prefix, max_length=1024, cache_file_name=None):
    hf_dataset = Dataset.from_list(data)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        labels = []
        for text, ids in zip(examples["text"], tokenized["input_ids"]):
            lbl = ids.copy()
            idx = text.find(assistant_prefix)

            if idx != -1:
                prefix_ids = tokenizer(
                    text[:idx],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )["input_ids"]

                if len(prefix_ids) < len(lbl):
                    lbl[:len(prefix_ids)] = [IGNORE_INDEX] * len(prefix_ids)

            pad_id = tokenizer.pad_token_id
            lbl = [IGNORE_INDEX if token_id == pad_id else l for token_id, l in zip(ids, lbl)]
            labels.append(lbl)

        tokenized["labels"] = labels
        return tokenized

    dataset = hf_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=hf_dataset.column_names,
        cache_file_name=cache_file_name,
    )

    return dataset


def train_single_model(model_cfg, train_raw, val_raw, script_name, full_description):
    print(f"\n{'='*60}")
    print(f"Starting training: {model_cfg['name']}")
    print(f"{'='*60}")

    model = None
    trainer = None
    tokenizer = None

    try:
        tmpl = CHAT_TEMPLATES[model_cfg["template_type"]]
        template_str = tmpl["prompt_template"]
        assistant_prefix = tmpl["assistant_prefix"]

        train_data = apply_template(train_raw, template_str)
        val_data = apply_template(val_raw, template_str)

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"

        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_version = "v4_fixed_padding"
        train_cache = f"{CACHE_DIR}/{cache_version}_train_{model_cfg['output_prefix']}_{script_name}.arrow"
        val_cache = f"{CACHE_DIR}/{cache_version}_val_{model_cfg['output_prefix']}_{script_name}.arrow"

        train_ds = preprocess_dataset(train_data, tokenizer, assistant_prefix, cache_file_name=train_cache)
        val_ds = preprocess_dataset(val_data, tokenizer, assistant_prefix, cache_file_name=val_cache)
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

        cleanup_memory()
        print_gpu_memory()

        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_dir = f"{RESULTS_DIR}/{model_cfg['output_prefix']}_{script_name}"
        os.makedirs(output_dir, exist_ok=True)

        cleanup_incomplete_checkpoints(output_dir)

        if check_completion_flag(model_cfg, COMPLETION_FLAG_DIR):
            print(f"Training already completed for {model_cfg['name']}, skipping...")
            return True

        checkpoint_info = find_latest_checkpoint(output_dir)
        resume_from_checkpoint = None

        if checkpoint_info:
            resume_from_checkpoint, latest_step = checkpoint_info
            print(f"Resuming from checkpoint-{latest_step}: {resume_from_checkpoint}")
            cleanup_memory()
            print_gpu_memory()
        else:
            print(f"Starting training from scratch")

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"],
            device_map=model_cfg["device_map"],
            torch_dtype=model_cfg["torch_dtype"],
        )

        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model, "gradient_checkpointing"):
                model.gradient_checkpointing = True
        except Exception as e:
            print(f"Could not enable gradient checkpointing: {e}")

        print_gpu_memory()

        training_args_dict = TRAINING_ARGS.copy()

        if training_args_dict.get("optim") == "adamw_8bit" and not BITSANDBYTES_AVAILABLE:
            print("bitsandbytes not available. Falling back to adamw_torch")
            training_args_dict["optim"] = "adamw_torch"

        if "dataloader_num_workers" not in training_args_dict:
            training_args_dict["dataloader_num_workers"] = 0
        elif training_args_dict.get("dataloader_num_workers", 4) > 2:
            training_args_dict["dataloader_num_workers"] = 2

        os.makedirs(RUNS_DIR, exist_ok=True)
        training_args_dict["logging_dir"] = f"{RUNS_DIR}/{model_cfg['output_prefix']}_{script_name}"

        args = TrainingArguments(
            output_dir=output_dir,
            run_name=model_cfg["output_prefix"],
            **training_args_dict,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        cleanup_memory()
        print_gpu_memory()

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print(f"Saving model to: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        create_completion_flag(model_cfg, output_dir, COMPLETION_FLAG_DIR)

        print(f"Training completed: {model_cfg['name']}")
        print_gpu_memory()

        return True

    except Exception as e:
        print(f"\nError training {model_cfg['name']}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

        cleanup_memory()
        print_gpu_memory()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cleanup_memory()
    print_gpu_memory()

    script_name = os.path.basename(sys.argv[0]).replace(".py", "")

    full_description = load_complexity_description(COMPLEXITY_DESC_PATH)
    if not full_description:
        print("Cannot proceed without complexity description")
        exit(1)

    print(f"\nLoading dataset: {DATASET_ID}")
    raw_ds = load_dataset(DATASET_ID, data_dir=DATASET_DIR, split="train")
    print(f"Dataset loaded: {len(raw_ds)} examples")

    all_data = create_all_combinations_data(raw_ds, full_description)

    random.seed(42)
    random.shuffle(all_data)
    split = int(0.9 * len(all_data))
    train_raw, val_raw = all_data[:split], all_data[split:]
    print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")

    total_models = len(SINGLE_MODEL_CONFIGS)
    successful_models = []
    failed_models = []

    for idx, model_cfg in enumerate(SINGLE_MODEL_CONFIGS, 1):
        print(f"\n{'='*60}")
        print(f"Model {idx}/{total_models}: {model_cfg['name']}")
        print(f"{'='*60}")

        success = train_single_model(model_cfg, train_raw, val_raw, script_name, full_description)

        if success:
            successful_models.append(model_cfg['name'])
        else:
            failed_models.append(model_cfg['name'])

        cleanup_memory()
        print_gpu_memory()

    print_training_summary(successful_models, failed_models, total_models)