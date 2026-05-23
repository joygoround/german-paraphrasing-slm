# eval_prompt.py
#
# Prompts used across generation and LLM-as-a-Judge evaluation.
#
# Generation prompts (fed to fine-tuned SLMs via eval_generate.py):
#   GENERATION_SYSTEM_PROMPT  — system role
#   GENERATION_USER_PROMPT    — user turn template
#     Placeholders: {{TARGET_LEVEL_LABEL}}, {{LEVEL_DESCRIPTION}}, {{SOURCE}}
#
# LLM Judge prompts (used by eval_llm_judge_tgi.py):
#   LLM_JUDGE_SYSTEM_PROMPT   — shared system role for all judges
#   Per-sample (5 dimensions): Coherency, Factuality, ComplexityAppropriateness,
#                               FactualIntegrity, FluencyNaturalness
#   Cross-level (1 prompt):    LevelMonotonicity → Monotonicity Score + Anchor Score


# ─────────────────────────────────────────────────────────────
# Generation prompts
# ─────────────────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = (
    "You are a helpful assistant trained by Liquid AI for German paraphrasing."
)

# The full complexity description (complexity_desc.txt) is prepended at runtime
# by eval_utils._build_user_content(), matching the fine-tuning prompt format.
GENERATION_USER_PROMPT = (
    "### AUFGABE:\n"
    "Paraphrasiere den folgenden Quelltext auf {{TARGET_LEVEL_LABEL}}.\n\n"
    "### QUELLTEXT:\n"
    "{{SOURCE}}\n\n"
    "### ZIEL-LEVEL:\n"
    "{{TARGET_LEVEL_LABEL}} — {{LEVEL_DESCRIPTION}}\n\n"
    "### AUSGABEFORMAT:\n"
    "Gib NUR den paraphrasierten deutschen Text aus."
)


# ─────────────────────────────────────────────────────────────
# LLM Judge — shared system prompt
# ─────────────────────────────────────────────────────────────

LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for German text complexity adaptation tasks.
Your evaluations are used for scientific research. Be precise, evidence-based, and consistent.
Always quote specific phrases from the text to support your judgements.
You evaluate on a 1–5 integer scale where 1 = very poor and 5 = excellent."""


# ─────────────────────────────────────────────────────────────
# Per-sample judge prompts (G-Eval pattern: CoT → Score)
# Placeholders: {{SOURCE}}, {{TARGET_LEVEL}}, {{LEVEL_DESCRIPTION}}, {{OUTPUT}}
# {{EXPECTED_OUTPUT}} used only by FactualIntegrity
# ─────────────────────────────────────────────────────────────

LLM_JUDGE_COHERENCY_PROMPT = """You are evaluating the COHERENCY of a German text paraphrase.

## Task Context
A model was asked to rewrite the following German source text at complexity {{TARGET_LEVEL}} ({{LEVEL_DESCRIPTION}}).

## Source Text
{{SOURCE}}

## Model Output
{{OUTPUT}}

## Evaluation Criteria
Coherency measures whether the paraphrase reads as a logically connected, well-organized whole:
- Ideas flow naturally from sentence to sentence
- Discourse connectives are used correctly (e.g., "jedoch", "daher", "außerdem")
- Sentence splitting or restructuring has not broken coreference or logical sequence
- The paraphrase does not contain non-sequiturs or abrupt topic jumps
Note: Do NOT penalize for simplification or vocabulary changes — only evaluate logical flow and organization.

## Scoring Rubric
- 5: Perfectly coherent; ideas flow naturally with no structural issues
- 4: Mostly coherent; one minor flow issue that does not impede understanding
- 3: Partially coherent; some notable structural gaps or awkward transitions
- 2: Mostly incoherent; frequent structural problems that hinder reading
- 1: Incoherent; ideas are disconnected or the text cannot be followed

## Your Evaluation
Step 1 — Identify any points where the logical flow breaks down. Quote the specific phrases.
Step 2 — Assess whether transitions between sentences are clear and appropriate for the target level.
Step 3 — Determine whether any sentence splitting has broken coreference or discourse structure.
Step 4 — Based on your analysis, assign a score.

Score: <integer 1-5>"""


LLM_JUDGE_FACTUALITY_PROMPT = """You are evaluating the FACTUALITY of a German text paraphrase.

## Task Context
A model was asked to rewrite the following German source text at complexity {{TARGET_LEVEL}} ({{LEVEL_DESCRIPTION}}).

## Source Text
{{SOURCE}}

## Model Output
{{OUTPUT}}

## Evaluation Criteria
Factuality measures whether the paraphrase introduces incorrect factual claims:
- Named entities (persons, organizations, places) must be preserved correctly
- Numbers, dates, statistics, and quantities must match the source
- Events and their relationships must not be distorted
- No facts may be invented that are not present in the source
Note: Rephrasing and simplification are expected. Only penalize factual inaccuracies, not style changes.

## Scoring Rubric
- 5: All facts are accurate; no errors, distortions, or hallucinations
- 4: Negligible imprecision (e.g., minor rounding) with no meaningful impact
- 3: One factual error or distortion that changes the meaning of a claim
- 2: Multiple factual errors or one serious distortion of a key fact
- 1: Severe factual corruption; the output misrepresents the source content

## Your Evaluation
Step 1 — List all verifiable facts in the source (names, numbers, dates, events).
Step 2 — Check each against the model output. Quote both source and output for any discrepancies.
Step 3 — Classify each discrepancy: distortion / omission / hallucination / acceptable rephrasing.
Step 4 — Based on your analysis, assign a score.

Score: <integer 1-5>"""


LLM_JUDGE_COMPLEXITY_APPROPRIATENESS_PROMPT = """You are evaluating the COMPLEXITY APPROPRIATENESS of a German text paraphrase.

## Task Context
A model was asked to rewrite the following German source text at complexity {{TARGET_LEVEL}} ({{LEVEL_DESCRIPTION}}).

## Source Text
{{SOURCE}}

## Model Output
{{OUTPUT}}

## Level Reference
- Level 1 (Leichte Sprache Plus): Very short sentences, only high-frequency words, no subordinate clauses, no abbreviations or metaphors
- Level 2 (Einfaches Deutsch): Short sentences, A2/B1 vocabulary, minimal subordination, no cultural idioms
- Level 3 (Alltagssprache): Clear standard sentences, general-public vocabulary, moderate complexity
- Level 4 (Gehobene Alltagssprache): Varied vocabulary, complex syntax, nuanced expressions
- Level 5 (Fachsprache/Akademisch): Nominalization, specialized terminology, multi-clause academic sentences

## Evaluation Criteria
Complexity Appropriateness measures whether the output matches the target level:
- Vocabulary register (too formal / too colloquial / appropriate)
- Sentence length and syntactic complexity (too complex / too simple / appropriate)
- Density of subordinate clauses and nominal phrases
- Presence or absence of technical jargon as expected by the level

## Scoring Rubric
- 5: Output perfectly matches the target level across vocabulary, syntax, and register
- 4: Mostly appropriate; one or two minor deviations from the target level
- 3: Partially appropriate; noticeable mismatches in either vocabulary or syntax
- 2: Mostly inappropriate; the output is clearly too simple or too complex for the target level
- 1: Completely wrong level; the output could not be mistaken for the target level

## Your Evaluation
Step 1 — Assess the vocabulary register. Quote 2–3 words or phrases as evidence.
Step 2 — Assess sentence length and syntactic complexity. Note average sentence length and any subordinate clauses.
Step 3 — Note any specific features that match or violate the target level definition.
Step 4 — Based on your analysis, assign a score.

Score: <integer 1-5>"""


LLM_JUDGE_FACTUAL_INTEGRITY_PROMPT = """You are evaluating the FACTUAL INTEGRITY of a German text paraphrase.

## Task Context
A model was asked to rewrite the following German source text at complexity {{TARGET_LEVEL}} ({{LEVEL_DESCRIPTION}}).

## Source Text
{{SOURCE}}

## Reference Output (human-authored simplification at this level)
{{EXPECTED_OUTPUT}}

## Model Output
{{OUTPUT}}

## Evaluation Criteria
Factual Integrity is a fine-grained audit of content preservation and information change:

**Information loss** — facts present in the source that are missing from the output
**Acceptable additions** — explanatory paraphrases or definitions that clarify the source (appropriate for lower levels)
**Unacceptable additions** — invented facts, embellishments, or claims not grounded in the source
Note: The reference output is provided for orientation only. Do NOT penalize the model for differing from it stylistically.
Compare the MODEL OUTPUT against the SOURCE TEXT, not against the reference.

## Scoring Rubric
- 5: All key facts preserved; any additions are purely explanatory and grounded in the source
- 4: All key facts preserved; one minor acceptable addition or one trivial omission of a secondary detail
- 3: One key fact is lost or one unacceptable addition is present
- 2: Multiple key facts are lost or distorted, or multiple unacceptable additions
- 1: Severe loss of factual content or fabricated information that changes the message of the text

## Your Evaluation
Step 1 — List the key facts in the source (entities, numbers, causal claims).
Step 2 — For each, check: preserved / lost / distorted. Quote evidence from the model output.
Step 3 — Check for additions: classify as explanatory (acceptable) or fabricated (unacceptable). Quote evidence.
Step 4 — Based on your analysis, assign a score.

Score: <integer 1-5>"""


LLM_JUDGE_FLUENCY_NATURALNESS_PROMPT = """You are evaluating the FLUENCY AND NATURALNESS of a German text paraphrase.

## Task Context
A model was asked to rewrite the following German source text at complexity {{TARGET_LEVEL}} ({{LEVEL_DESCRIPTION}}).

## Source Text
{{SOURCE}}

## Model Output
{{OUTPUT}}

## Evaluation Criteria
Fluency and Naturalness measures whether the output reads as idiomatic, grammatically correct German
appropriate for a native speaker at the target level:
- Grammatical correctness: case agreement, gender agreement, verb position (V2 in main clauses, verb-final in subordinate clauses)
- Idiomatic phrasing: no calqued structures, no unnatural word order
- Register consistency: phrasing matches the expected register for the target level throughout
- Absence of repetition artifacts or incomplete sentences
Note: Do NOT penalize for simplification, vocabulary changes, or content differences — only evaluate linguistic quality.

## Scoring Rubric
- 5: Perfectly fluent; reads as natural German with no awkward phrasings
- 4: Mostly fluent; one minor awkwardness that does not impede reading
- 3: Partially fluent; several unnatural phrasings or one clear grammatical error
- 2: Mostly unnatural; frequent grammatical errors or consistently awkward phrasing
- 1: Unnatural throughout; grammatical errors or non-German constructions dominate the text

## Your Evaluation
Step 1 — Check verb position in main and subordinate clauses. Quote any violations.
Step 2 — Check grammatical agreement (case, gender, number). Quote any errors.
Step 3 — Identify any unnatural or calqued phrasings. Quote examples.
Step 4 — Assess register consistency across the whole output.
Step 5 — Based on your analysis, assign a score.

Score: <integer 1-5>"""


# ─────────────────────────────────────────────────────────────
# Cross-level judge (one call per source text, all 5 outputs)
# Placeholders: {{TASK_DESCRIPTION}}, {{SOURCE}}, {{OUTPUT_L1}}..{{OUTPUT_L5}}
# ─────────────────────────────────────────────────────────────

LLM_JUDGE_LEVEL_MONOTONICITY_PROMPT = """You are evaluating LEVEL MONOTONICITY across five complexity-graded German paraphrases.

## Task Context
{{TASK_DESCRIPTION}}

## Source Text
{{SOURCE}}

## Five Outputs for the Same Source

**Level 1 (Leichte Sprache Plus — very simple, high-frequency words only)**:
{{OUTPUT_L1}}

**Level 2 (Einfaches Deutsch — A2/B1 vocabulary, short sentences)**:
{{OUTPUT_L2}}

**Level 3 (Alltagssprache — standard German, general public)**:
{{OUTPUT_L3}}

**Level 4 (Gehobene Alltagssprache — varied vocab, complex syntax)**:
{{OUTPUT_L4}}

**Level 5 (Fachsprache/Akademisch — nominalization, specialized terms)**:
{{OUTPUT_L5}}

## Level Complexity Markers

| Level | Key Markers |
|-------|-------------|
| L1 | ≤10-word sentences, no subordination, no abbreviations |
| L2 | Short sentences, basic vocab, minimal subordination |
| L3 | Standard sentences, moderate subordination, general vocab |
| L4 | Long sentences, complex syntax, varied/nuanced vocab |
| L5 | Nominalization, multi-clause structures, domain terminology |

## Evaluation Criteria
Monotonicity requires that linguistic complexity increases consistently from L1 to L5.
Assess four adjacent transitions: L1→L2, L2→L3, L3→L4, L4→L5.
For each transition, determine: Increasing / Flat / Reversed.
Then evaluate two sub-scores: Monotonicity and Anchor Quality.

## Scoring Rubrics

**Monotonicity Score (1–5)**:
- 5: All four transitions are Increasing
- 4: Three transitions Increasing, one Flat
- 3: Two transitions Increasing, others Flat or one Reversed
- 2: Only one Increasing transition, or two Reversed
- 1: No Increasing transitions or multiple Reversals

**Anchor Score (1–5)** — quality of the extreme levels only:
- 5: L1 is clearly minimal (very short, basic); L5 is clearly maximal (nominal, technical)
- 4: Both anchors mostly meet their requirements with minor lapses
- 3: One anchor meets requirements; the other partially fails
- 2: Both anchors partially fail their requirements
- 1: Both anchors fail — L1 is not simple enough and/or L5 is not complex enough

## Your Evaluation
Step 1 — For each adjacent pair (L1→L2, L2→L3, L3→L4, L4→L5): state Increasing / Flat / Reversed and quote one piece of evidence from both levels.
Step 2 — Identify any Reversed or Flat transitions and explain the cause.
Step 3 — Evaluate L1 anchor: does it meet minimum simplicity requirements? Quote evidence.
Step 4 — Evaluate L5 anchor: does it meet maximum complexity requirements? Quote evidence.
Step 5 — Assign the two scores.

Monotonicity Score: <integer 1-5>
Anchor Score: <integer 1-5>"""