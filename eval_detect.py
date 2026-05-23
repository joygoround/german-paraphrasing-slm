# eval_detect.py
#
# Computes DETECT scores (Simplicity, Meaning Preservation, Fluency, Total)
# from a pre-generated outputs CSV produced by eval_generate.py.
#
# DETECT must be installed separately:
#   https://github.com/ZurichNLP/DETECT
# Update DETECT_PATH below to point to your local clone.
#
# Usage:
#   python eval_detect.py <outputs_csv>

import sys
import pandas as pd
from tqdm.auto import tqdm
from eval_utils import get_dynamic_filename, save_experiment_metadata, setup_reproducibility

DETECT_PATH = ''  # path to your local DETECT clone (https://github.com/ZurichNLP/DETECT)
sys.path.insert(0, DETECT_PATH)
from lens.detect_score import DETECT


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_detect.py <outputs_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    setup_reproducibility()
    output_file = get_dynamic_filename(extension=".csv")
    save_experiment_metadata(output_file)

    print(f"Loading outputs from: {input_csv}")
    df_in = pd.read_csv(input_csv)

    required = {"Model", "Sample_ID", "Level", "Source", "Reference", "Output"}
    missing = required - set(df_in.columns)
    if missing:
        print(f"Missing columns in input CSV: {missing}")
        sys.exit(1)

    print("Initialising DETECT scorer...")
    detect_scorer = DETECT(rescale=True)

    results = []

    for (model_name, level), group in tqdm(
        df_in.groupby(["Model", "Level"]), desc="Model x Level"
    ):
        srcs = group["Source"].tolist()
        hyps = group["Output"].tolist()
        refs = [[r] for r in group["Reference"].tolist()]

        try:
            scores = detect_scorer.score(srcs, hyps, refs, batch_size=8, devices=[0])
            n = len(scores)
            results.append({
                "Model":        model_name,
                "Level":        level,
                "Simplicity":   round(sum(s["simplicity"]           for s in scores) / n, 4),
                "Meaning":      round(sum(s["meaning_preservation"]  for s in scores) / n, 4),
                "Fluency":      round(sum(s["fluency"]               for s in scores) / n, 4),
                "DETECT_Total": round(sum(s["total"]                 for s in scores) / n, 4),
            })
        except Exception as e:
            print(f"DETECT error ({model_name} L{level}): {e}")

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print("\n" + "=" * 50 + "\nDETECT SUMMARY\n" + "=" * 50)
        print(df_out.groupby("Model")[["DETECT_Total", "Simplicity", "Meaning", "Fluency"]].mean().to_string())
    else:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()