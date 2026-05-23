# eval_reference.py
#
# Reference-based metrics: SARI, BERTScore F1, chrF
# Reads pre-generated outputs CSV from eval_generate.py.
#
# Usage:
#   python eval_reference.py <outputs_csv>

import sys
import pandas as pd
from tqdm.auto import tqdm
from evaluate import load as load_metric
from bert_score import score as bert_score_func
import sacrebleu

from eval_config import CONFIG
from eval_utils import get_dynamic_filename, save_experiment_metadata, setup_reproducibility


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_reference.py <outputs_csv>")
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

    try:
        sari_metric = load_metric("sari")
    except Exception as e:
        print(f"Failed to load SARI metric: {e}")
        sys.exit(1)

    results = []

    for (model_name, level), group in tqdm(
        df_in.groupby(["Model", "Level"]), desc="Model x Level"
    ):
        srcs = group["Source"].tolist()
        hyps = group["Output"].tolist()
        refs = [[r] for r in group["Reference"].tolist()]

        # SARI (corpus-level scalar)
        try:
            sari_score = sari_metric.compute(
                sources=srcs, predictions=hyps, references=refs
            )["sari"]
        except Exception as e:
            print(f"SARI error ({model_name} L{level}): {e}")
            sari_score = None

        # BERTScore (per-sample F1, German encoder)
        try:
            _, _, F1 = bert_score_func(
                hyps, [r[0] for r in refs],
                lang="de",
                model_type=CONFIG.get("bert_model"),
                device="cuda",
                verbose=False,
            )
        except Exception:
            print(f"Falling back to default BERTScore model (L{level})...")
            try:
                _, _, F1 = bert_score_func(
                    hyps, [r[0] for r in refs],
                    lang="de",
                    device="cuda",
                    verbose=False,
                )
            except Exception as e2:
                print(f"BERTScore error ({model_name} L{level}): {e2}")
                F1 = [None] * len(hyps)

        f1_list = F1.tolist() if hasattr(F1, "tolist") else F1

        # chrF (corpus-level, via sacrebleu)
        try:
            chrf_score = sacrebleu.corpus_chrf(
                hyps, [[r[0] for r in refs]]
            ).score
        except Exception as e:
            print(f"chrF error ({model_name} L{level}): {e}")
            chrf_score = None

        for i, row in enumerate(group.itertuples(index=False)):
            results.append({
                "Model":        model_name,
                "Sample_ID":    row.Sample_ID,
                "Level":        level,
                "SARI_Corpus":  round(sari_score, 4)  if sari_score  is not None else None,
                "BERTScore_F1": round(f1_list[i], 4)  if f1_list[i]  is not None else None,
                "chrF":         round(chrf_score, 4)   if chrf_score  is not None else None,
            })

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        summary = df_out.groupby(["Model", "Level"])[
            ["SARI_Corpus", "BERTScore_F1", "chrF"]
        ].mean()
        print("\n" + "=" * 55)
        print("          REFERENCE-BASED METRICS SUMMARY")
        print("=" * 55)
        print(summary.to_string())
    else:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()