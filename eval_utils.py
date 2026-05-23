# eval_utils.py
import torch
from eval_config import CONFIG, LEVEL_GUIDES
from eval_prompt import GENERATION_SYSTEM_PROMPT, GENERATION_USER_PROMPT
import textstat
import random
import numpy as np
import os
import sys
from datetime import datetime

_COMPLEXITY_DESC_PATH = os.path.join(os.path.dirname(__file__), "complexity_desc.txt")
try:
    with open(_COMPLEXITY_DESC_PATH, "r", encoding="utf-8") as _f:
        COMPLEXITY_FULL_DESC = _f.read().strip()
except FileNotFoundError:
    print(f"complexity_desc.txt not found at {_COMPLEXITY_DESC_PATH}. User prompt will omit it.")
    COMPLEXITY_FULL_DESC = ""


def _render(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def _build_user_content(level: int, source: str) -> str:
    level_label = f"Level {level}"
    level_desc = LEVEL_GUIDES[str(level)]
    task_block = _render(
        GENERATION_USER_PROMPT,
        TARGET_LEVEL_LABEL=level_label,
        LEVEL_DESCRIPTION=level_desc,
        SOURCE=source,
    )
    if COMPLEXITY_FULL_DESC:
        return f"{COMPLEXITY_FULL_DESC}\n\n{task_block}"
    return task_block


def generate_paraphrase(model, tokenizer, input_text: str, level: int) -> str:
    """
    Generate a complexity-adapted paraphrase using zero-shot prompting.
    Chat format matches fine-tuning: system prompt + complexity_desc.txt + task block.
    """
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(level, input_text)},
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=CONFIG["temperature"],
            top_k=CONFIG["top_k"],
            top_p=CONFIG["top_p"],
            repetition_penalty=CONFIG["repetition_penalty"],
            max_new_tokens=CONFIG["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_text = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    for prefix in ["Paraphrase:", "### PARAPHRASE (NUR TEXT) ###", "### AUSGABE ###"]:
        gen_text = gen_text.replace(prefix, "").strip()

    return gen_text.strip('"').strip()


def get_dynamic_filename(extension=".csv"):
    script_name = os.path.basename(sys.argv[0]).replace(".py", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"results_{script_name}_{timestamp}{extension}"


def save_experiment_metadata(results_filename):
    metadata_filename = results_filename.replace(".csv", "_metadata.txt")
    with open(metadata_filename, "w", encoding="utf-8") as f:
        f.write(f"SCRIPT: {sys.argv[0]}\n")
        f.write(f"DATE: {datetime.now()}\n")
        f.write("-" * 40 + "\n")
        f.write(f"CONFIG: {CONFIG}\n")
        f.write(f"PROMPT (system): {GENERATION_SYSTEM_PROMPT}\n")


def setup_reproducibility():
    seed = CONFIG["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_readability_scores(text):
    textstat.set_lang('de')
    if not text or not text.strip():
        return {"WSTF": 0.0, "LIX": 0.0, "FRE_DE": 0.0}
    return {
        "WSTF": round(textstat.wiener_sachtextformel(text, variant=1), 2),
        "LIX": round(textstat.lix(text), 2),
        "FRE_DE": round(textstat.flesch_reading_ease(text), 2),
    }