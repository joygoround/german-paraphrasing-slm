# eval_surface.py
#
# Surface and structural metrics (all reference-free):
#   Readability : WSTF, LIX, FRE_DE
#   Structural  : AvgSentLen, AvgWordLen, PropLongWords, LengthRatio, SentSplitRate
#
# Reads pre-generated outputs CSV from eval_generate.py.
#
# Usage:
#   python eval_surface.py <outputs_csv>

import sys
import re
import pandas as pd
from tqdm.auto import tqdm

from eval_utils import get_dynamic_filename, save_experiment_metadata, setup_reproducibility, get_readability_scores


def _sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in parts if s]


def get_structural_scores(source: str, output: str) -> dict:
    src_sents  = _sentences(source)
    out_sents  = _sentences(output)
    out_words  = output.split()

    n_out_words = len(out_words)
    n_src_words = len(source.split())
    n_out_sents = max(len(out_sents), 1)

    return {
        "AvgSentLen":    round(n_out_words / n_out_sents, 2),
        "AvgWordLen":    round(sum(len(w) for w in out_words) / max(n_out_words, 1), 2),
        "PropLongWords": round(sum(1 for w in out_words if len(w) > 6) / max(n_out_words, 1), 4),
        "LengthRatio":   round(n_out_words / max(n_src_words, 1), 4),
        "SentSplitRate": round(len(out_sents) / max(len(src_sents), 1), 4),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_surface.py <outputs_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    setup_reproducibility()
    output_file = get_dynamic_filename(extension=".csv")
    save_experiment_metadata(output_file)

    print(f"Loading outputs from: {input_csv}")
    df_in = pd.read_csv(input_csv)

    required = {"Model", "Sample_ID", "Level", "Source", "Output"}
    missing = required - set(df_in.columns)
    if missing:
        print(f"Missing columns in input CSV: {missing}")
        sys.exit(1)

    results = []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Surface metrics"):
        readability = get_readability_scores(row["Output"])
        structural  = get_structural_scores(row["Source"], row["Output"])

        results.append({
            "Model":     row["Model"],
            "Sample_ID": row["Sample_ID"],
            "Level":     row["Level"],
            "WSTF":      readability["WSTF"],
            "LIX":       readability["LIX"],
            "FRE_DE":    readability["FRE_DE"],
            **structural,
        })

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        metric_cols = ["WSTF", "LIX", "FRE_DE",
                       "AvgSentLen", "AvgWordLen", "PropLongWords",
                       "LengthRatio", "SentSplitRate"]
        summary = df_out.groupby(["Model", "Level"])[metric_cols].mean()
        print("\n" + "=" * 65)
        print("             SURFACE & STRUCTURAL METRICS SUMMARY")
        print("=" * 65)
        print(summary.to_string())
    else:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()