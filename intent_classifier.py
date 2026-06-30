"""
intent_classifier.py
====================
Classifies a user query as one of:
    "dataset"  - needs soil moisture data analysis
    "chat"     - general conversation / greetings
"""

import re
import requests


# ============================================================================
# MINIMAL FAST PRE-FILTERS
# (Only handle cases where we are 100% confident without the LLM)
# ============================================================================

# Words that unambiguously indicate a dataset action
_STRONG_DATA = {
    'mean', 'average', 'minimum', 'maximum', 'trend', 'slope',
    'compare', 'comparison', 'algorithm', 'retrieval', 'methodology',
    'rmse', 'bias', 'validation', 'calibration', 'backscatter',
}

# Regions list (loaded lazily)
def _get_regions():
    try:
        from Config import VALID_REGIONS
        return set(VALID_REGIONS) | {'india', 'national', 'country'}
    except Exception:
        return {'india'}

_YEAR_RE = re.compile(r'\b20\d{2}\b')

_GREETINGS = {
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
    'how are you', 'who are you', 'what can you do', 'help', 'what is your name',
}


def _fast_classify(query: str):
    """
    Returns 'dataset' or 'chat' OR None (= needs LLM).

    Rules (all must be high-confidence):
      - Short greeting → 'chat'
      - Has region + year + data action → 'dataset'
      - Has strong data word alone → 'dataset'
      - Everything else → None (LLM decides)
    """
    q     = query.lower().strip()
    words = set(q.split())
    has_strong_data = any(kw in q for kw in _STRONG_DATA)

    # Greeting / chat check
    is_chat = False
    if len(words) <= 8 and any(g in q for g in _GREETINGS):
        is_chat = True
    elif any(phrase in q for phrase in ['what is your name', 'who are you', 'what can you do', 'how are you']):
        is_chat = True

    if is_chat and not has_strong_data:
        return 'chat'

    regions     = _get_regions()
    has_region  = any(r in q for r in regions)
    has_year    = bool(_YEAR_RE.search(q))

    # Clear dataset: region + year + data action
    if has_region and has_year and has_strong_data:
        return 'dataset'

    # Strong data signal alone → dataset
    if has_strong_data:
        return 'dataset'

    # Let the LLM handle ambiguous cases
    return None


# ============================================================================
# LLM CLASSIFIER  (handles ambiguous, mixed, and nuanced queries)
# ============================================================================

_LLM_PROMPT = (
    "You are a routing classifier for a soil moisture analysis system.\n"
    "\n"
    "Classify the user query into EXACTLY ONE of these two categories:\n"
    "\n"
    "  chat\n"
    "    - Greetings, small talk, 'who are you', 'what can you do', off-topic.\n"
    "\n"
    "  dataset\n"
    "    - Wants soil moisture statistics, maps, trends, comparisons, regional\n"
    "      analysis, numerical data for a specific region and/or time period.\n"
    "    - Also use this for technical questions about algorithms, sensors, or\n"
    "      validation metrics (RMSE, bias) even without a specific region.\n"
    "    - Examples:\n"
    "        'What is the mean soil moisture in Rajasthan in June 2020?'\n"
    "        'Compare Rajasthan and Gujarat in 2021'\n"
    "        'Show moisture trend in Punjab monsoon 2022'\n"
    "        'What is the RMSE for SMAP validation?'\n"
    "\n"
    "Respond with ONLY ONE WORD - no punctuation, no explanation:\n"
    "chat | dataset\n"
    "\n"
    "User query: {query}\n"
    "\n"
    "Category:"
)




def _llm_classify(
    query: str,
    ollama_url: str,
    ollama_model: str,
    timeout: int,
) -> str:
    """Call Ollama to classify the query intent."""
    try:
        resp = requests.post(
            ollama_url,
            json={
                "model" : ollama_model,
                "prompt": _LLM_PROMPT.format(query=query),
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p"      : 0.1,
                    "num_ctx"    : 2048,
                    "num_predict": 8,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip().lower()
        raw_clean = re.sub(r"[^a-z\s]", "", raw).strip()
        print("LLM CLASSIFIER OUTPUT:", raw_clean)

        # Accept exact match first
        if raw_clean in ("chat", "dataset", "literature", "both"):
            return raw_clean
        # LLM sometimes outputs two words joined or with extra text — 
        # pick the FIRST valid intent word found
        for token in ("both", "literature", "dataset", "chat"):
            if token in raw_clean:
                print(f"  (parsed token: {token})")
                return token

    except Exception as e:
        print("LLM CLASSIFIER ERROR:", e)

    return "dataset"   # safe fallback — try to run dataset analysis


# ============================================================================
# PUBLIC API
# ============================================================================

def classify_query_intent(
    query: str,
    ollama_url: str   = "http://localhost:11434/api/generate",
    ollama_model: str = "qwen2.5:3b",
    timeout: int      = 30,
) -> str:
    """
    Classify a user query intent.

    Returns: "dataset" | "chat"
    If LLM returns "literature" or "both" (old prompt), they are
    collapsed to "dataset" so the app always attempts data analysis.
    """
    fast = _fast_classify(query)
    if fast is not None:
        print("FINAL INTENT (FAST):", fast)
        return fast

    result = _llm_classify(
        query        = query,
        ollama_url   = ollama_url,
        ollama_model = ollama_model,
        timeout      = timeout,
    )
    # Collapse any leftover "literature" / "both" to "dataset"
    if result in ("literature", "both"):
        result = "dataset"
    print("FINAL INTENT (LLM):", result)
    return result


# ============================================================================
# DEBUG HELPER
# ============================================================================

def explain_classification(query: str, intent: str) -> str:
    q = query.lower()
    if intent == "dataset":
        hits = [kw for kw in _STRONG_DATA if kw in q]
        return f"📊 DATASET — Soil moisture data/statistics query. Signals: {hits[:4]}"
    return "💬 CHAT — General conversation."