from models.sentiment import analyze_sentiment
from models.keywords import extract_keywords
from models.linguistic import extract_features
from models.emotions import detect_emotions
from models.embeddings import get_embedding_summary
from pipeline.llm_reasoner import run_llm_analysis


def run_pipeline(text: str) -> dict:
    print(f"[pipeline] Starting analysis on {len(text)} chars...")

    results = {}
    errors  = []

    # ── LAYER 1 ──────────────────────────────────
    try:
        print("[pipeline] sentiment...")
        results["sentiment"] = analyze_sentiment(text)
    except Exception as e:
        errors.append(f"sentiment: {e}")
        results["sentiment"] = None

    try:
        print("[pipeline] keywords...")
        results["keywords"] = extract_keywords(text)
    except Exception as e:
        errors.append(f"keywords: {e}")
        results["keywords"] = None

    try:
        print("[pipeline] linguistic...")
        results["linguistic"] = extract_features(text)
    except Exception as e:
        errors.append(f"linguistic: {e}")
        results["linguistic"] = None

    # ── LAYER 2 ──────────────────────────────────
    try:
        print("[pipeline] emotions...")
        results["emotions"] = detect_emotions(text)
    except Exception as e:
        errors.append(f"emotions: {e}")
        results["emotions"] = None

    try:
        print("[pipeline] embeddings...")
        results["embeddings"] = get_embedding_summary(text)
    except Exception as e:
        errors.append(f"embeddings: {e}")
        results["embeddings"] = None

    # ── LAYER 3 — LLM ────────────────────────────
    try:
        print("[pipeline] LLM reasoning...")
        analysis = run_llm_analysis(text, results)
    except Exception as e:
        errors.append(f"llm: {e}")
        analysis = {"error": str(e)}

    print(f"[pipeline] Done! Errors: {errors if errors else 'none'}")

    return {
        "status":  "ok" if not errors else "partial",
        "errors":  errors,
        "text_length": len(text),
        "signals": results,
        "analysis": analysis
    }