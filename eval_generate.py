# eval_generate_outputs.py
#
# Generates complexity-graded paraphrases for all models and saves them to CSV.
# All downstream eval scripts (eval_sari_bertscore.py, eval_detect.py,
# eval_llm_judge.py) read from this output instead of regenerating.
#
# Usage:
#   python eval_generate_outputs.py
#
# Output CSV columns:
#   Model, Sample_ID, Level, Source, Reference, Output

import os
import gc
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_config import MODEL_PATHS
from eval_utils import (
    generate_paraphrase,
    get_dynamic_filename,
    save_experiment_metadata,
    setup_reproducibility,
)


def main():
    setup_reproducibility()
    output_file = get_dynamic_filename(extension=".csv")
    save_experiment_metadata(output_file)

    print("Loading test dataset...")
    ds = load_dataset("tum-nlp/German4All-Corpus", data_dir="corrected", split="test")

    results = []

    for model_name, path in MODEL_PATHS.items():
        print(f"\nLoading model: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, device_map="cuda", dtype=torch.bfloat16, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model.eval()

            for level in range(1, 6):
                level_key = f"cl_{level}"
                print(f"  Level {level} ...")

                for entry in tqdm(ds, desc=f"{model_name} L{level}"):
                    gen_text = generate_paraphrase(
                        model, tokenizer, entry["text"], level
                    )
                    results.append({
                        "Model":     model_name,
                        "Sample_ID": entry["id"],
                        "Level":     level,
                        "Source":    entry["text"],
                        "Reference": entry[level_key],
                        "Output":    gen_text,
                    })

                pd.DataFrame(results).to_csv(
                    output_file.replace(".csv", "_checkpoint.csv"), index=False
                )

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error for {model_name}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nOutputs saved to: {output_file}")
        print(f"  {len(df)} rows  |  {df['Model'].nunique()} models  |  {df['Level'].nunique()} levels")
    else:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()