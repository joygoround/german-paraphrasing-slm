# -*- coding: utf-8 -*-
import os
import random
import sys
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from config_train import (
    MODEL_CONFIGS_MULTITASK as SINGLE_MODEL_CONFIGS,
    CHAT_TEMPLATES,
    DATASET_ID,
    DATASET_DIR,
    SRC_COL,
    TARGET_COL_PREFIX,
    COMPLEXITY_LEVELS,
    LAMBDA_COMPLEXITY,
    TRAINING_ARGS_MULTITASK as TRAINING_ARGS,
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
NUM_LEVELS = len(COMPLEXITY_LEVELS)


def create_multitask_data(dataset_split, full_description):
    data = []
    all_sources = ["ORIGINAL"] + [f"Level {l}" for l in COMPLEXITY_LEVELS]

    for ex in dataset_split:
        text_map = {"ORIGINAL": ex[SRC_COL]}
        for l in COMPLEXITY_LEVELS:
            col = f"{TARGET_COL_PREFIX}{l}"
            if col in ex and ex[col]:
                text_map[f"Level {l}"] = ex[col]

        for target_l in COMPLEXITY_LEVELS:
            tgt_key = f"Level {target_l}"
            if tgt_key not in text_map:
                continue
            for src_key in all_sources:
                if src_key not in text_map or src_key == tgt_key:
                    continue
                prompt = (
                    f"{full_description}\n"
                    f"Paraphrasiere den folgenden Text von {src_key} "
                    f"auf Level {target_l}.\n"
                    f"Text: {text_map[src_key]}"
                )
                data.append({
                    "prompt": prompt,
                    "target_text": text_map[tgt_key],
                    "complexity_label": target_l - 1,
                })

    print(f"Generated {len(data)} training samples")
    return data


class ParaphraseWithComplexityHead(nn.Module):
    """
    LM backbone with an auxiliary linear classification head for complexity level prediction.
    The head is trained jointly with the LM objective and removed at inference time.
    """
    def __init__(self, model_name, num_levels=5, device_map="auto", torch_dtype=None):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            output_hidden_states=True,
        )
        hidden_size = self.lm.config.hidden_size
        self.complexity_head = nn.Linear(hidden_size, num_levels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.config = self.lm.config

    def tie_weights(self):
        if hasattr(self.lm, "tie_weights"):
            self.lm.tie_weights()

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.lm.save_pretrained(save_directory, **kwargs)
        complexity_head_path = os.path.join(save_directory, "complexity_head.pt")
        torch.save(self.complexity_head.state_dict(), complexity_head_path)
        self.config.save_pretrained(save_directory)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.lm, 'gradient_checkpointing_enable'):
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.lm, 'gradient_checkpointing_disable'):
            self.lm.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, labels=None, paraphrase_end_indices=None, complexity_labels=None):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lm_loss = outputs.loss

        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            lm_loss = torch.tensor(0.0, device=lm_loss.device, requires_grad=True)

        total_loss = lm_loss

        if complexity_labels is not None and paraphrase_end_indices is not None:
            last_hidden = outputs.hidden_states[-1]
            indices = paraphrase_end_indices.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
            batch_embeddings = torch.gather(last_hidden, 1, indices).squeeze(1)

            logits = self.complexity_head(batch_embeddings)
            complexity_loss = self.loss_fn(logits, complexity_labels)
            total_loss = lm_loss + (LAMBDA_COMPLEXITY * complexity_loss)

            return {
                "loss": total_loss,
                "logits": logits,
                "labels": complexity_labels,
            }

        return {"loss": total_loss, "logits": outputs.logits}


def apply_template(data, template):
    return [
        {
            "text": template.format(prompt=d["prompt"], target_text=d["target_text"]),
            "complexity_label": d["complexity_label"],
        }
        for d in data
    ]


def preprocess_dataset(data, tokenizer, assistant_prefix, max_length=1024, cache_file_name=None):
    hf_ds = Dataset.from_list(data)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        all_labels, all_end_indices = [], []

        for i in range(len(tokenized["input_ids"])):
            input_ids = tokenized["input_ids"][i]
            offsets = tokenized["offset_mapping"][i]
            text = examples["text"][i]
            a_idx = text.find(assistant_prefix)
            label = list(input_ids)
            if a_idx != -1:
                for j, (start, end) in enumerate(offsets):
                    if start < a_idx:
                        label[j] = IGNORE_INDEX
                    else:
                        break
            all_labels.append(label)
            all_end_indices.append(len(label) - 1)

        tokenized["labels"] = all_labels
        tokenized["paraphrase_end_indices"] = all_end_indices
        tokenized["complexity_labels"] = examples["complexity_label"]
        del tokenized["offset_mapping"]
        return tokenized

    dataset = hf_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=hf_ds.column_names,
        desc="Tokenizing",
        cache_file_name=cache_file_name,
    )
    dataset.set_format("torch")
    return dataset


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    def force_flatten(data):
        flat_list = []
        if isinstance(data, (list, tuple, np.ndarray)):
            for item in data:
                flat_list.extend(force_flatten(item))
        else:
            flat_list.append(data)
        return flat_list

    flat_preds = np.array(force_flatten(preds))
    flat_labels = np.array(force_flatten(labels))

    num_samples = len(flat_preds)

    if len(flat_labels) > num_samples:
        print(f"Warning: dimension mismatch. Preds: {len(flat_preds)}, Labels: {len(flat_labels)}")
        flat_labels = flat_labels[:num_samples]

    flat_preds = np.nan_to_num(flat_preds)

    if len(flat_preds.shape) > 1:
        flat_preds = np.argmax(flat_preds, axis=-1)

    acc = accuracy_score(flat_labels, flat_preds)
    return {"complexity_accuracy": acc}


def create_multitask_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(features):
        max_batch_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for f in features:
            ids = f["input_ids"].tolist() if isinstance(f["input_ids"], torch.Tensor) else f["input_ids"]
            mask = f["attention_mask"].tolist() if isinstance(f["attention_mask"], torch.Tensor) else f["attention_mask"]
            lbls = f["labels"].tolist() if isinstance(f["labels"], torch.Tensor) else f["labels"]
            pad_len = max_batch_len - len(ids)
            batch["input_ids"].append(torch.tensor(ids + [pad_id] * pad_len))
            batch["attention_mask"].append(torch.tensor(mask + [0] * pad_len))
            batch["labels"].append(torch.tensor(lbls + [IGNORE_INDEX] * pad_len))

        res = {k: torch.stack(v) for k, v in batch.items()}
        res["complexity_labels"] = torch.tensor(
            [f["complexity_labels"] for f in features], dtype=torch.long
        )
        res["paraphrase_end_indices"] = torch.tensor(
            [min(f["paraphrase_end_indices"], max_batch_len - 1) for f in features],
            dtype=torch.long,
        )
        return res

    return collate


class SaveLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"\n[CHECKPOINT] Saved at step {state.global_step}")
        if model is not None and hasattr(model, 'lm') and self.tokenizer is not None:
            try:
                model.lm.config.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
            except Exception as e:
                print(f"Failed to save config/tokenizer: {e}")


def train_single_model(model_cfg, train_raw, val_raw, script_name, full_description):
    print(f"\n{'='*60}")
    print(f"Starting multitask training: {model_cfg['name']}")
    print(f"  Lambda Complexity: {LAMBDA_COMPLEXITY}")
    print(f"{'='*60}")

    try:
        tmpl = CHAT_TEMPLATES[model_cfg["template_type"]]
        template_str = tmpl["prompt_template"]
        assistant_prefix = tmpl["assistant_prefix"]

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        os.makedirs(CACHE_DIR, exist_ok=True)
        train_cache = f"{CACHE_DIR}/v3_train_{model_cfg['output_prefix']}_{script_name}.arrow"
        val_cache = f"{CACHE_DIR}/v3_val_{model_cfg['output_prefix']}_{script_name}.arrow"

        train_ds = preprocess_dataset(
            apply_template(train_raw, template_str), tokenizer, assistant_prefix,
            cache_file_name=train_cache,
        )
        val_ds = preprocess_dataset(
            apply_template(val_raw, template_str), tokenizer, assistant_prefix,
            cache_file_name=val_cache,
        )

        MAX_EVAL_SAMPLES = 500
        if len(val_ds) > MAX_EVAL_SAMPLES:
            val_ds = val_ds.select(range(MAX_EVAL_SAMPLES))

        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

        cleanup_memory()
        print_gpu_memory()

        model = ParaphraseWithComplexityHead(
            model_cfg["model_id"], NUM_LEVELS,
            device_map=model_cfg["device_map"],
            torch_dtype=model_cfg["torch_dtype"],
        )

        if hasattr(model.lm, "gradient_checkpointing_enable"):
            model.lm.gradient_checkpointing_enable()

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
        else:
            print(f"Starting training from scratch")

        training_args_dict = TRAINING_ARGS.copy()
        training_args_dict["label_names"] = ["complexity_labels"]

        if training_args_dict.get("optim") == "adamw_8bit" and not BITSANDBYTES_AVAILABLE:
            print("bitsandbytes not available. Falling back to adamw_torch")
            training_args_dict["optim"] = "adamw_torch"

        os.makedirs(RUNS_DIR, exist_ok=True)
        training_args_dict["logging_dir"] = f"{RUNS_DIR}/{model_cfg['output_prefix']}_{script_name}"

        args = TrainingArguments(
            output_dir=output_dir,
            run_name=model_cfg["output_prefix"],
            **training_args_dict,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=create_multitask_collate_fn(tokenizer),
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[SaveLoggingCallback(tokenizer=tokenizer)],
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        final_model_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        model.lm.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        torch.save(model.complexity_head.state_dict(), os.path.join(final_model_dir, "complexity_head.pt"))
        print(f"Model saved to: {final_model_dir}")

        stats_path = os.path.join(output_dir, "training_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Training completed for: {model_cfg['name']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total steps: {trainer.state.global_step}\n")
            f.write(f"Total epochs: {trainer.state.epoch}\n")
            if trainer.state.log_history:
                f.write(f"Final loss: {trainer.state.log_history[-1].get('loss', 'N/A')}\n")

        create_completion_flag(model_cfg, output_dir, COMPLETION_FLAG_DIR)
        print(f"Training completed: {model_cfg['name']}")
        print_gpu_memory()

        del model, trainer
        cleanup_memory()

        return True

    except Exception as e:
        print(f"\nError training {model_cfg['name']}: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    script_name = os.path.basename(sys.argv[0]).replace(".py", "")

    full_description = load_complexity_description(COMPLEXITY_DESC_PATH)
    if not full_description:
        print("Cannot proceed without complexity description")
        exit(1)

    print(f"\nLoading dataset: {DATASET_ID}")
    raw_ds = load_dataset(DATASET_ID, data_dir=DATASET_DIR, split="train")
    print(f"Dataset loaded: {len(raw_ds)} examples")

    print(f"\nCreating multitask training samples...")
    all_data = create_multitask_data(raw_ds, full_description)

    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(0.9 * len(all_data))
    train_raw, val_raw = all_data[:split_idx], all_data[split_idx:]
    print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")

    total_models = len(SINGLE_MODEL_CONFIGS)
    successful, failed = [], []

    for idx, model_cfg in enumerate(SINGLE_MODEL_CONFIGS, 1):
        print(f"\n{'='*60}")
        print(f"Model {idx}/{total_models}: {model_cfg['name']}")
        print(f"{'='*60}")

        if train_single_model(model_cfg, train_raw, val_raw, script_name, full_description):
            successful.append(model_cfg['name'])
        else:
            failed.append(model_cfg['name'])

        cleanup_memory()

    print_training_summary(successful, failed, total_models)