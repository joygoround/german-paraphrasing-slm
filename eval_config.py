# eval_config.py

CONFIG = {
    "seed": 29,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 0.1,
    "repetition_penalty": 1.05,
    "max_new_tokens": 512,
    "bert_model": "dbmdz/bert-base-german-cased",
}

# Model keys must match the "Model Key" column in Table 4.1 of the thesis.
MODEL_PATHS = {
    "LFM2-Baseline":      "LiquidAI/LFM2-1.2B",
    "LFM2.5-Baseline":    "LiquidAI/LFM2.5-1.2B-Instruct",
    "LFM2-AllComb":       "joygoround/2_single_allcom-lfm2",
    "LFM2-Contrastive":   "joygoround/3_constrative-lfm2",
    "LFM2-Multitask":     "joygoround/4_multitask-lfm2",
    "LFM2.5-AllComb":     "joygoround/2_allcomb-lfm2.5",
    "LFM2.5-Contrastive": "joygoround/3_constrative-lfm2.5",
    "LFM2.5-Multitask":   "joygoround/4_multitask-lfm2.5",
}

LEVEL_GUIDES = {
    "1": (
        "Level 1: Leichte Sprache Plus. "
        "Zielgruppe: Menschen mit Leseschwierigkeiten oder Deutschlerner. "
        "Charakteristika: Sehr kurze Sätze, nur häufig verwendete Wörter, direkte Ansprache. "
        "WICHTIG: Vermeide Abkürzungen, Metaphern, Ironie oder komplexe Satzgefüge."
    ),
    "2": (
        "Level 2: Einfaches Deutsch für Anfänger. "
        "Zielgruppe: Nicht-Muttersprachler mit Grundkenntnissen. "
        "Charakteristika: Einfacher Satzbau, Grundwortschatz, Fokus auf die wichtigsten Informationen. "
        "WICHTIG: Vermeide kulturspezifische Ausdrücke."
    ),
    "3": (
        "Level 3: Standardsprache (Alltagssprache). "
        "Zielgruppe: Allgemeines Publikum. "
        "Charakteristika: Klare, gut strukturierte Sätze. Fokus auf Verständlichkeit. "
        "WICHTIG: Vermeide Fachbegriffe weitgehend."
    ),
    "4": (
        "Level 4: Gehobene Alltagssprache. "
        "Zielgruppe: Regelmäßige Leser mit gutem Sprachverständnis. "
        "Charakteristika: Abwechslungsreicher Wortschatz, komplexe Satzstrukturen. "
        "WICHTIG: Fachbegriffe sind erlaubt, sollten aber erklärt werden."
    ),
    "5": (
        "Level 5: Akademische Sprache. "
        "Zielgruppe: Akademiker und Experten. "
        "Charakteristika: Komplexe Satzgefüge, spezialisierte Terminologie, "
        "häufige Verwendung von Fachbegriffen ohne zusätzliche Erklärungen."
    ),
}