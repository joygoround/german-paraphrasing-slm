import os
import gc
import glob
import re
import shutil
import torch
from datetime import datetime


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def cleanup_incomplete_checkpoints(output_dir):
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    removed_count = 0
    for ckpt in checkpoint_dirs:
        if not os.path.exists(os.path.join(ckpt, "trainer_state.json")):
            shutil.rmtree(ckpt)
            removed_count += 1
    if removed_count > 0:
        print(f"Cleaned up {removed_count} incomplete checkpoint(s)")


def find_latest_checkpoint(output_dir):
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None

    def get_step_number(path):
        match = re.search(r'checkpoint-(\d+)$', path)
        return int(match.group(1)) if match else 0

    latest = max(checkpoint_dirs, key=get_step_number)
    step = get_step_number(latest)
    if os.path.exists(os.path.join(latest, "trainer_state.json")):
        return latest, step
    return None


def create_completion_flag(model_config, output_dir, flag_dir):
    os.makedirs(flag_dir, exist_ok=True)
    flag_path = os.path.join(flag_dir, f"training_completed_{model_config['output_prefix']}.flag")
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_config['name']}\n")
        f.write(f"Model ID: {model_config['model_id']}\n")
        f.write(f"Output Directory: {output_dir}\n")


def check_completion_flag(model_config, flag_dir):
    flag_path = os.path.join(flag_dir, f"training_completed_{model_config['output_prefix']}.flag")
    return os.path.exists(flag_path)


def load_complexity_description(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Missing complexity description: {file_path}")
        return ""


def print_training_summary(successful_models, failed_models, total_models):
    print(f"\n{'='*60}")
    print(f"Training Pipeline Summary")
    print(f"{'='*60}")
    print(f"Successful: {len(successful_models)}/{total_models}")
    for name in successful_models:
        print(f"  - {name}")
    if failed_models:
        print(f"\nFailed: {len(failed_models)}/{total_models}")
        for name in failed_models:
            print(f"  - {name}")
    print(f"\n{'='*60}")
    if len(successful_models) == total_models:
        print("All models trained successfully.")
    else:
        print(f"Training completed with {len(failed_models)} failure(s).")
    print(f"{'='*60}\n")