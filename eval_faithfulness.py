# eval_faithfulness.py
#
# Faithfulness metric: Named Entity (NE) Retention Rate
# Measures what proportion of PER, ORG, LOC entities in the source
# are preserved in the model output.
#
# Uses spaCy de_core_news_lg. Install with:
#   pip install spacy && python -m spacy download de_core_news_lg
#
# Reads pre-generated outputs CSV from eval_generate.py.
#
# Usage:
#   python eval_faithfulness.py <outputs_csv>

import sys
import spacy
import pandas as pd
from tqdm.auto import tqdm

from eval_utils import get_dynamic_filename, save_experiment_metadata, setup_reproducibility

NE_LABELS = {"PER", "ORG", "LOC"}


def load_nlp():
    try:
        return spacy.load("de_core_news_lg")
    except OSError:
        print("spaCy model 'de_core_news_lg' not found.")
        print("Run: python -m spacy download de_core_news_lg")
        sys.exit(1)


def ne_retention_rate(nlp, source: str, output: str) -> float | None:
    """
    Returns the proportion of source NEs (PER, ORG, LOC) whose text
    appears in the output. Returns None if the source has no NEs.
    """
    src_doc = nlp(source)
    src_nes = {ent.text.lower() for ent in src_doc.ents if ent.label_ in NE_LABELS}

    if not src_nes:
        return None

    output_lower = output.lower()
    retained = sum(1 for ne in src_nes if ne in output_lower)
    return round(retained / len(src_nes), 4)


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_faithfulness.py <outputs_csv>")
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

    print("Loading spaCy de_core_news_lg...")
    nlp = load_nlp()

    results = []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="NE retention"):
        rate = ne_retention_rate(nlp, row["Source"], row["Output"])
        results.append({
            "Model":            row["Model"],
            "Sample_ID":        row["Sample_ID"],
            "Level":            row["Level"],
            "NE_RetentionRate": rate,
        })

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        # Exclude samples with no source NEs from the summary average
        summary = (
            df_out.dropna(subset=["NE_RetentionRate"])
            .groupby(["Model", "Level"])["NE_RetentionRate"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "NE_RetentionRate_Mean", "count": "NE_Samples"})
        )
        print("\n" + "=" * 55)
        print("          NE RETENTION RATE SUMMARY")
        print("  (NE_Samples = samples with at least one NE in source)")
        print("=" * 55)
        print(summary.to_string())
    else:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()