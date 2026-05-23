# eval_llm_judge_tgi.py
#
# LLM-as-a-Judge evaluation using local judge models served via a
# Text Generation Inference (TGI) server (OpenAI-compatible API).
#
# Each judge runs all 7 evaluation dimensions per output:
#   Per-sample (5): Coherency, Factuality, ComplexityAppropriateness,
#                   FactualIntegrity, FluencyNaturalness
#   Cross-level (2): LevelMonotonicity → Monotonicity Score + Anchor Score
#
# Reads pre-generated outputs from eval_generate.py.
#
# Usage:
#   python eval_llm_judge_tgi.py <outputs_csv> [--resume <checkpoint_csv>]
#
# Required environment variables (.env):
#   TGI_BASE_URL  — base URL of your TGI server, e.g. http://myserver:8080/v1
#   TGI_API_KEY   — optional; defaults to "tgi"

import argparse
import os
import re
import sys
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

from eval_config import LEVEL_GUIDES
from eval_utils import get_dynamic_filename, save_experiment_metadata, setup_reproducibility
from eval_prompt import (
    LLM_JUDGE_SYSTEM_PROMPT,
    LLM_JUDGE_COHERENCY_PROMPT,
    LLM_JUDGE_FACTUALITY_PROMPT,
    LLM_JUDGE_COMPLEXITY_APPROPRIATENESS_PROMPT,
    LLM_JUDGE_FACTUAL_INTEGRITY_PROMPT,
    LLM_JUDGE_FLUENCY_NATURALNESS_PROMPT,
    LLM_JUDGE_LEVEL_MONOTONICITY_PROMPT,
)

# Compact cross-level prompt: the eval_prompt version asks for step-by-step analysis,
# which makes smaller instruction-tuned judges sometimes omit the final score lines.
_LLM_JUDGE_LEVEL_MONOTONICITY_PROMPT_COMPACT = """You are evaluating LEVEL MONOTONICITY across five complexity-graded German paraphrases.

## Task Context
{{TASK_DESCRIPTION}}

## Source Text
{{SOURCE}}

## Five Outputs for the Same Source
L1:
{{OUTPUT_L1}}

L2:
{{OUTPUT_L2}}

L3:
{{OUTPUT_L3}}

L4:
{{OUTPUT_L4}}

L5:
{{OUTPUT_L5}}

## Scoring
Monotonicity Score (1–5): how consistently complexity increases from L1→L5.
Anchor Score (1–5): how well L1 is clearly simple and L5 clearly complex.
"""

# Ensure progress / warnings appear promptly even under nohup.
try:
    sys.stdout.reconfigure(line_buffering=True)  # py3.7+
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# =============================================================================
# JUDGE MODEL REGISTRY
# model_id must match exactly what your TGI server was started with.
# Both judges share the same TGI client (same base URL).
# =============================================================================
JUDGE_CONFIGS = [
    {
        "name": "Gemma3-12B",
        "model_id": "google/gemma-3-12b-it",
        "extra": {},
    },
    # {
    #     "name": "Qwen3-14B",
    #     "model_id": "Qwen/Qwen3-14B",
    #     "extra": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    #     "strip_thinking": True,
    # },
]

JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256

TASK_DESCRIPTION = (
    "Rewrite a German news text into one of five complexity levels: "
    "Level 1 (Leichte Sprache Plus) through Level 5 (Fachsprache/Akademisch). "
    "All original facts must be preserved. Only the paraphrased German text is output."
)


# =============================================================================
# CLIENT
# =============================================================================
def _build_tgi_client() -> OpenAI:
    base_url = os.getenv("TGI_BASE_URL")
    if not base_url:
        raise EnvironmentError(
            "TGI_BASE_URL is not set. Add it to your .env file, "
            "e.g.  TGI_BASE_URL=http://myserver:8080/v1"
        )
    api_key = os.getenv("TGI_API_KEY", "tgi")
    return OpenAI(base_url=base_url, api_key=api_key)


# =============================================================================
# JUDGE CALL
# =============================================================================
def call_judge(client: OpenAI, model_id: str, user_prompt: str, extra: dict,
               strip_thinking: bool = False,
               max_tokens: int | None = None) -> str:
    response = client.chat.completions.create(
        model=model_id,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=(JUDGE_MAX_TOKENS if max_tokens is None else max_tokens),
        messages=[
            {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        **extra,
    )
    content = response.choices[0].message.content or ""
    if strip_thinking:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


# =============================================================================
# PROMPT RENDERING
# =============================================================================
def render(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    t = str(text)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 20] + "\n...[TRUNCATED]..."


# =============================================================================
# SCORE PARSING
# =============================================================================
# Many models return markdown like "**Score:** 4" or variants like "Score - 4".
# Accept a wider range of formats while still restricting to integer 1–5.
_SCORE_RE = re.compile(r"\bScore\b\s*[:：\-]?\s*[^0-9]*([1-5])\b", re.IGNORECASE)
_SCORE_ALT_RE = re.compile(
    r"\b(?:Rating|Bewertung|Punktzahl)\b\s*[:：\-]?\s*[^0-9]*([1-5])\b",
    re.IGNORECASE,
)
_SCORE_FALLBACK_RE = re.compile(r"\b([1-5])\s*/\s*5\b")
_SCORE_LAST_INT_RE = re.compile(r"([1-5])\s*$")
_FIRST_LINE_INT_RE = re.compile(r"^\s*(?:\*\*)?(?:Score|Rating|Bewertung|Punktzahl)?(?:\*\*)?\s*[:：\-]?\s*([1-5])\b", re.IGNORECASE)
_MONO_RE = re.compile(
    r"\b(?:Level\s+)?Monotonicity\b(?:\s*Score)?\s*[:：\-]?\s*[^0-9]*([1-5])\b",
    re.IGNORECASE,
)
# German variants show up surprisingly often (e.g., "Monotonie", "Anker").
_MONO_DE_RE = re.compile(
    r"\bMonoton(?:ie|izit(?:ä|a)t)\b(?:\s*Score)?\s*[:：\-]?\s*[^0-9]*([1-5])\b",
    re.IGNORECASE,
)
# Models often respond with "Anchor Quality: 4" / "AnchorQuality Score: 4" even though we ask for
# "Anchor Score". Be permissive: Anchor, Anchor Quality, AnchorQuality, with/without "Score".
_ANCH_RE = re.compile(
    r"\bAnchor(?:\s*Quality|Quality)?\b(?:\s*Score)?\s*[:：\-]?\s*[^0-9]*([1-5])\b",
    re.IGNORECASE,
)
_ANCH_DE_RE = re.compile(
    r"\bAnker(?:\s*(?:Qualit(?:ä|a)t))?\b(?:\s*Score)?\s*[:：\-]?\s*[^0-9]*([1-5])\b",
    re.IGNORECASE,
)
_INT_1_TO_5_RE = re.compile(r"\b([1-5])\b")


def parse_score(text: str) -> float | None:
    if not text:
        return None
    # Fast path: the first line often contains the score (sometimes without ':' or with markdown).
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    m = _FIRST_LINE_INT_RE.search(first_line)
    if m:
        return float(m.group(1))
    for pat in (_SCORE_RE, _SCORE_ALT_RE):
        m = pat.findall(text)
        if m:
            return float(m[-1])
    m = _SCORE_FALLBACK_RE.search(text)
    if m:
        return float(m.group(1))
    m = _SCORE_LAST_INT_RE.search(text.strip())
    if m:
        return float(m.group(1))
    print(f"[WARN] parse_score failed: {text[:120]!r}")
    return None


def parse_monotonicity_scores(text: str) -> tuple[float | None, float | None]:
    mono = _MONO_RE.search(text) or _MONO_DE_RE.search(text)
    anch = _ANCH_RE.search(text) or _ANCH_DE_RE.search(text)
    mono_score = float(mono.group(1)) if mono else None
    anch_score = float(anch.group(1)) if anch else None
    # Fallback: if the model returned ONLY two integers, treat them as (mono, anchor).
    # This recovers cases like:
    #   "Monotonicity: 4\\nAnchor: 5" (already handled) or just "4\\n5".
    if mono_score is None or anch_score is None:
        ints = [int(x) for x in _INT_1_TO_5_RE.findall(text or "")]
        # Keep order but de-duplicate consecutive repeats (models sometimes echo "5 5").
        cleaned = []
        for x in ints:
            if not cleaned or cleaned[-1] != x:
                cleaned.append(x)
        if len(cleaned) == 2:
            mono_score = float(cleaned[0])
            anch_score = float(cleaned[1])
    if mono_score is None or anch_score is None:
        # Keep this compact but actionable; missing values otherwise silently become NaN in the CSV.
        print(
            f"[WARN] parse_monotonicity_scores failed (mono={mono_score}, anchor={anch_score}): {text[:220]!r}",
            flush=True,
        )
    return (mono_score, anch_score)


# =============================================================================
# PER-SAMPLE JUDGE RUNNERS
# =============================================================================
def run_per_sample_judges(client: OpenAI, model_id: str, extra: dict,
                          source: str, level: int,
                          reference: str, output: str,
                          strip_thinking: bool = False) -> dict:
    level_desc = LEVEL_GUIDES[str(level)]

    def _score(template, **kw):
        prompt = render(template, **kw) + (
            "\n\n"
            "CRITICAL OUTPUT FORMAT:\n"
            "- First line MUST be exactly: Score: <integer 1-5>\n"
            "- After that, at most 2 short sentences of justification.\n"
            "- Do NOT use markdown for the score line.\n"
        )
        return parse_score(call_judge(client, model_id, prompt, extra, strip_thinking, max_tokens=128))

    return {
        "J_Coherency": _score(
            LLM_JUDGE_COHERENCY_PROMPT,
            SOURCE=source, TARGET_LEVEL=level, LEVEL_DESCRIPTION=level_desc, OUTPUT=output,
        ),
        "J_Factuality": _score(
            LLM_JUDGE_FACTUALITY_PROMPT,
            SOURCE=source, TARGET_LEVEL=level, LEVEL_DESCRIPTION=level_desc, OUTPUT=output,
        ),
        "J_ComplexityAppropriateness": _score(
            LLM_JUDGE_COMPLEXITY_APPROPRIATENESS_PROMPT,
            SOURCE=source, TARGET_LEVEL=level, LEVEL_DESCRIPTION=level_desc, OUTPUT=output,
        ),
        "J_FactualIntegrity": _score(
            LLM_JUDGE_FACTUAL_INTEGRITY_PROMPT,
            SOURCE=source, TARGET_LEVEL=level, LEVEL_DESCRIPTION=level_desc,
            EXPECTED_OUTPUT=reference, OUTPUT=output,
        ),
        "J_FluencyNaturalness": _score(
            LLM_JUDGE_FLUENCY_NATURALNESS_PROMPT,
            SOURCE=source, TARGET_LEVEL=level, LEVEL_DESCRIPTION=level_desc, OUTPUT=output,
        ),
    }


def run_monotonicity_judge(client: OpenAI, model_id: str, extra: dict,
                           source: str, outputs_by_level: dict,
                           strip_thinking: bool = False) -> tuple[float | None, float | None, str]:
    # Keep this extremely short to fit within max_model_len=2048.
    format_guard = (
        "\n\nCRITICAL OUTPUT FORMAT:\n"
        "- Output EXACTLY two lines and nothing else.\n"
        "- Line 1 MUST be: Monotonicity Score: <integer 1-5>\n"
        "- Line 2 MUST be: Anchor Score: <integer 1-5>\n"
        "- No bullet points, no analysis, no markdown.\n"
    )

    def _prompt(src_chars: int, out_chars: int) -> str:
        return (
            render(
                _LLM_JUDGE_LEVEL_MONOTONICITY_PROMPT_COMPACT,
                TASK_DESCRIPTION=TASK_DESCRIPTION,
                SOURCE=_truncate(source, src_chars),
                OUTPUT_L1=_truncate(outputs_by_level.get(1, ""), out_chars),
                OUTPUT_L2=_truncate(outputs_by_level.get(2, ""), out_chars),
                OUTPUT_L3=_truncate(outputs_by_level.get(3, ""), out_chars),
                OUTPUT_L4=_truncate(outputs_by_level.get(4, ""), out_chars),
                OUTPUT_L5=_truncate(outputs_by_level.get(5, ""), out_chars),
            )
            + format_guard
        )

    # First try (moderate truncation). If the server returns context-length 400,
    # retry once with more aggressive truncation and fewer output tokens.
    try:
        response = call_judge(
            client,
            model_id,
            _prompt(src_chars=1000, out_chars=600),
            extra,
            strip_thinking,
            max_tokens=32,
        )
    except Exception as e:
        if "maximum context length" not in str(e):
            raise
        response = call_judge(
            client,
            model_id,
            _prompt(src_chars=700, out_chars=420),
            extra,
            strip_thinking,
            max_tokens=32,
        )
    mono_score, anch_score = parse_monotonicity_scores(response)
    return mono_score, anch_score, response


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation via a local TGI server (no OpenAI tokens)"
    )
    parser.add_argument("outputs_csv", help="Path to the generated outputs CSV")
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT_CSV",
        default=None,
        help="Path to a checkpoint CSV from a previous run. Already-completed "
             "(JudgeModel, SLMModel, Sample_ID) triples are skipped.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Build TGI client (shared by both judges — same server, different model_id)
    # -------------------------------------------------------------------------
    client = _build_tgi_client()
    print(f"TGI server: {os.getenv('TGI_BASE_URL')}")

    # Used for filenames when starting fresh (before judge loop assigns judge_name).
    # If multiple judges are configured, keep the filename generic.
    default_judge_name = (
        JUDGE_CONFIGS[0]["name"]
        if len(JUDGE_CONFIGS) == 1 and "name" in JUDGE_CONFIGS[0]
        else "multi_judge"
    )

    # -------------------------------------------------------------------------
    # Load input
    # -------------------------------------------------------------------------
    setup_reproducibility()

    print(f"Loading outputs from: {args.outputs_csv}")
    df_in = pd.read_csv(args.outputs_csv)

    required = {"Model", "Sample_ID", "Level", "Source", "Reference", "Output"}
    missing = required - set(df_in.columns)
    if missing:
        print(f"Missing columns in input CSV: {missing}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Resume: load existing checkpoint rows and build skip-set
    # -------------------------------------------------------------------------
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        df_ckpt = pd.read_csv(args.resume)
        results = df_ckpt.to_dict("records")
        done_set = set(
            zip(df_ckpt["JudgeModel"], df_ckpt["SLMModel"], df_ckpt["Sample_ID"])
        )
        output_file = args.resume.replace("_checkpoint.csv", ".csv")
        checkpoint_file = args.resume
        print(f"   Loaded {len(results)} existing rows; {len(done_set)} completed (judge, model, sample) triples.")
    else:
        output_file = get_dynamic_filename(extension=".csv").replace(
            ".csv", f"_{default_judge_name}.csv"
        )
        checkpoint_file = output_file.replace(".csv", "_checkpoint.csv")
        save_experiment_metadata(output_file)
        results = []
        done_set = set()

    # -------------------------------------------------------------------------
    # Judge loop
    # -------------------------------------------------------------------------
    judge_cols = [
        "J_Coherency", "J_Factuality", "J_ComplexityAppropriateness",
        "J_FactualIntegrity", "J_FluencyNaturalness",
        "J_Monotonicity", "J_AnchorQuality",
    ]

    for judge_cfg in JUDGE_CONFIGS:
        judge_name = judge_cfg["name"]
        model_id = judge_cfg["model_id"]
        extra = judge_cfg["extra"]
        strip_thinking = judge_cfg.get("strip_thinking", False)

        print(f"\nJudge: {judge_name} ({model_id})")

        for slm_model, model_group in tqdm(df_in.groupby("Model"), desc=judge_name):
            for sample_id, sample_group in model_group.groupby("Sample_ID"):

                if (judge_name, slm_model, sample_id) in done_set:
                    continue

                source = sample_group["Source"].iloc[0]
                outputs_by_level = dict(zip(
                    sample_group["Level"].astype(int), sample_group["Output"]
                ))

                # Cross-level judge — once per source × SLM model
                mono_score, anch_score, mono_raw = run_monotonicity_judge(
                    client, model_id, extra, source, outputs_by_level, strip_thinking
                )

                # Per-sample judges — once per level
                for _, row in sample_group.iterrows():
                    level = int(row["Level"])
                    output = row["Output"]
                    reference = row["Reference"]

                    scores = run_per_sample_judges(
                        client, model_id, extra,
                        source, level, reference, output, strip_thinking
                    )

                    results.append({
                        "JudgeModel": judge_name,
                        "SLMModel": slm_model,
                        "Sample_ID": sample_id,
                        "Level": level,
                        **scores,
                        "J_Monotonicity": mono_score,
                        "J_AnchorQuality": anch_score,
                        "J_LevelMonotonicity_Raw": mono_raw,
                    })

                done_set.add((judge_name, slm_model, sample_id))

                # Checkpoint after each sample
                pd.DataFrame(results).to_csv(checkpoint_file, index=False)

    if not results:
        print("\nNo data collected.")
        return

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Summary: mean scores per JudgeModel × SLMModel × Level
    summary = df_out.groupby(["JudgeModel", "SLMModel", "Level"])[judge_cols].mean()
    print("\n" + "=" * 70)
    print("                     LLM JUDGE SUMMARY (TGI)")
    print("=" * 70)
    print(summary.to_string())

    # Agreement table: std across judge models (lower = more agreement)
    agreement = df_out.groupby(["SLMModel", "Level"])[judge_cols].std()
    print("\n" + "=" * 70)
    print("         INTER-JUDGE STD (lower = more agreement between judges)")
    print("=" * 70)
    print(agreement.to_string())


if __name__ == "__main__":
    main()
