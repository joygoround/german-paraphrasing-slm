# -*- coding: utf-8 -*-
import os
import random
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from config_train import (
    MODEL_CONFIGS_CONTRASTIVE as SINGLE_MODEL_CONFIGS,
    CHAT_TEMPLATES,
    DATASET_ID,
    DATASET_DIR,
    SRC_COL,
    TARGET_COL_PREFIX,
    COMPLEXITY_LEVELS,
    TRAINING_ARGS_CONTRASTIVE as TRAINING_ARGS,
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


class LevelContrastiveTrainer(Trainer):
    """
    Extends HuggingFace Trainer with an InfoNCE contrastive loss term.
    The joint loss is: (1 - alpha) * L_LM + alpha * L_InfoNCE
    """
    def __init__(self, *args, contrastive_weight=0.3, temperature=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.gen_losses = []
        self.contra_losses = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            output_hidden_states=True,
        )
        gen_loss = outputs.loss

        neg_ids = inputs.get("negative_input_ids")
        if neg_ids is not None and neg_ids.numel() > 0:
            contra_loss = self._compute_infonce_loss(model, outputs, neg_ids)
        else:
            contra_loss = torch.tensor(0.0, device=gen_loss.device)

        total_loss = (1 - self.contrastive_weight) * gen_loss + self.contrastive_weight * contra_loss

        if self.state.global_step % 100 == 0:
            self._log_loss_components(gen_loss, contra_loss, total_loss)

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_infonce_loss(self, model, outputs, neg_ids):
        # Anchor embedding: mean-pool last hidden state
        pos_emb = outputs.hidden_states[-1].mean(dim=1)

        batch_size, num_negs, seq_len = neg_ids.shape
        flat_neg_ids = neg_ids.view(-1, seq_len)

        with torch.no_grad():
            neg_outputs = model(
                input_ids=flat_neg_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            neg_embs = neg_outputs.hidden_states[-1].mean(dim=1)
            neg_embs = neg_embs.view(batch_size, num_negs, -1)

        pos_emb_norm = F.normalize(pos_emb, p=2, dim=-1)
        neg_emb_norm = F.normalize(neg_embs, p=2, dim=-1)

        logits = torch.bmm(
            pos_emb_norm.unsqueeze(1),
            neg_emb_norm.transpose(1, 2),
        ).squeeze(1) / self.temperature

        return torch.logsumexp(logits, dim=1).mean()

    def _log_loss_components(self, gen_loss, contra_loss, total_loss):
        self.gen_losses.append(gen_loss.item())
        self.contra_losses.append(contra_loss.item())

        recent_gen = sum(self.gen_losses[-10:]) / min(len(self.gen_losses), 10)
        recent_contra = sum(self.contra_losses[-10:]) / min(len(self.contra_losses), 10)

        print(f"\nStep {self.state.global_step} loss breakdown:")
        print(f"  Generation:   {gen_loss.item():.4f} (avg: {recent_gen:.4f})")
        print(f"  Contrastive:  {contra_loss.item():.4f} (avg: {recent_contra:.4f})")
        print(f"  Total:        {total_loss.item():.4f}")


def create_contrastive_data(dataset_split, full_description, tokenizer, max_length=512):
    """
    Build contrastive training samples. For each complexity level, the other
    four levels from the same source text serve as hard negatives.
    Only samples with all five cl_* columns present are included.
    """
    all_data = []
    skipped = 0

    for ex in dataset_split:
        text_map = {}
        for l in COMPLEXITY_LEVELS:
            col = f"{TARGET_COL_PREFIX}{l}"
            if col in ex and ex[col] and len(ex[col].strip()) > 10:
                text_map[l] = ex[col]

        if len(text_map) < 5:
            skipped += 1
            continue

        for tgt_lvl in COMPLEXITY_LEVELS:
            tgt_key = f"Level {tgt_lvl}"
            src_key = "ORIGINAL"

            prompt = (
                f"{full_description}\n\n"
                f"### AUFGABE:\n"
                f"Paraphrasiere den folgenden Quelltext von {src_key} auf {tgt_key}.\n\n"
                f"### QUELLTEXT:\n{ex[SRC_COL]}\n\n"
                f"### ZIEL-LEVEL:\n{tgt_key}\n\n"
                f"### AUSGABEFORMAT:\n"
                f"Gib NUR den paraphrasierten deutschen Text aus."
            )

            target_text = text_map[tgt_lvl]
            neg_texts = [text_map[l] for l in COMPLEXITY_LEVELS if l != tgt_lvl]

            neg_tokenized = tokenizer(
                neg_texts,
                padding='max_length',
                max_length=max_length // 4,  # 128 tokens; fixed [batch, 4, 128] shape
                truncation=True,
                return_tensors='pt',
            )

            all_data.append({
                "prompt": prompt,
                "target_text": target_text,
                "negative_input_ids": neg_tokenized["input_ids"].tolist(),
            })

    print(f"Generated {len(all_data)} contrastive samples (skipped {skipped} incomplete)")
    return all_data


def apply_template(data, template):
    return [
        {
            "text": template.format(prompt=d["prompt"], target_text=d["target_text"]),
            "negative_input_ids": d["negative_input_ids"],
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
        )

        labels = []
        for text, ids in zip(examples["text"], tokenized["input_ids"]):
            lbl = list(ids)
            idx = text.find(assistant_prefix)
            if idx != -1:
                prefix_ids = tokenizer(
                    text[:idx],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
                if len(prefix_ids) < len(lbl):
                    lbl[:len(prefix_ids)] = [IGNORE_INDEX] * len(prefix_ids)
            labels.append(lbl)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "negative_input_ids": examples["negative_input_ids"],
        }

    dataset = hf_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        cache_file_name=cache_file_name,
        desc="Tokenizing dataset",
    )

    if len(dataset) > 0 and "negative_input_ids" not in dataset.column_names:
        raise ValueError("negative_input_ids was removed during preprocessing")

    return dataset


class ContrastiveDataCollator:
    """
    Pads positive sequences and stacks negative token tensors into a fixed
    [batch, 4, seq_len] tensor for the contrastive loss.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids, padded_attention_mask, padded_labels = [], [], []

        for ids, mask, lbl in zip(input_ids, attention_mask, labels):
            padding_length = max_len - len(ids)

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(mask, torch.Tensor):
                mask = mask.tolist()
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.tolist()

            padded_input_ids.append(ids + [self.pad_token_id] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)
            padded_labels.append(lbl + [IGNORE_INDEX] * padding_length)

        negative_input_ids = []
        for f in features:
            neg_ids = f["negative_input_ids"]
            if isinstance(neg_ids, torch.Tensor):
                negative_input_ids.append(neg_ids)
            else:
                negative_input_ids.append(torch.tensor(neg_ids, dtype=torch.long))

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "negative_input_ids": torch.stack(negative_input_ids),
        }


def train_single_model(model_cfg, train_raw, script_name, full_description):
    print(f"\n{'='*60}")
    print(f"Starting contrastive training: {model_cfg['name']}")
    print(f"{'='*60}")

    try:
        tmpl = CHAT_TEMPLATES[model_cfg["template_type"]]
        template_str = tmpl["prompt_template"]
        assistant_prefix = tmpl["assistant_prefix"]

        train_data = apply_template(train_raw, template_str)

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_version = "v8_final_fix"
        train_cache = f"{CACHE_DIR}/{cache_version}_train_{model_cfg['output_prefix']}_{script_name}.arrow"

        train_ds = preprocess_dataset(train_data, tokenizer, assistant_prefix, cache_file_name=train_cache)
        print(f"Train samples: {len(train_ds)}")
        print(f"Dataset columns: {train_ds.column_names}")

        cleanup_memory()
        print_gpu_memory()

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"],
            device_map=model_cfg["device_map"],
            torch_dtype=model_cfg["torch_dtype"],
        )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

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
        os.makedirs(RUNS_DIR, exist_ok=True)
        training_args_dict["logging_dir"] = f"{RUNS_DIR}/{model_cfg['output_prefix']}_{script_name}"

        args = TrainingArguments(
            output_dir=output_dir,
            run_name=f"{model_cfg['output_prefix']}_contrastive",
            remove_unused_columns=False,
            **training_args_dict,
        )

        trainer = LevelContrastiveTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=ContrastiveDataCollator(tokenizer),
            contrastive_weight=0.3,
            temperature=0.07,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print(f"Saving model to: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        stats_path = f"{output_dir}/training_stats.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Training statistics for {model_cfg['name']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total steps: {trainer.state.global_step}\n")
            f.write(f"Final loss: {trainer.state.log_history[-1].get('loss', 'N/A')}\n\n")
            if trainer.gen_losses:
                f.write(f"Avg generation loss:   {sum(trainer.gen_losses) / len(trainer.gen_losses):.4f}\n")
            if trainer.contra_losses:
                f.write(f"Avg contrastive loss:  {sum(trainer.contra_losses) / len(trainer.contra_losses):.4f}\n")

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


def train_model():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    script_name = os.path.basename(sys.argv[0]).replace(".py", "")

    full_description = load_complexity_description(COMPLEXITY_DESC_PATH)
    if not full_description:
        print("Cannot proceed without complexity description")
        exit(1)

    print(f"\nLoading dataset: {DATASET_ID}")
    raw_ds = load_dataset(DATASET_ID, data_dir=DATASET_DIR, split="train")
    print(f"Dataset loaded: {len(raw_ds)} examples")

    tokenizer = AutoTokenizer.from_pretrained(SINGLE_MODEL_CONFIGS[0]["model_id"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    print(f"\nCreating contrastive training samples...")
    contrastive_raw = create_contrastive_data(raw_ds, full_description, tokenizer)

    total_models = len(SINGLE_MODEL_CONFIGS)
    successful_models = []
    failed_models = []

    for idx, model_cfg in enumerate(SINGLE_MODEL_CONFIGS, 1):
        print(f"\n{'='*60}")
        print(f"Model {idx}/{total_models}: {model_cfg['name']}")
        print(f"{'='*60}")

        success = train_single_model(model_cfg, contrastive_raw, script_name, full_description)

        if success:
            successful_models.append(model_cfg['name'])
        else:
            failed_models.append(model_cfg['name'])

        cleanup_memory()

    print_training_summary(successful_models, failed_models, total_models)


if __name__ == "__main__":
    train_model()