from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

print("[sentiment] Loading VADER...")
vader = SentimentIntensityAnalyzer()

print("[sentiment] Loading RoBERTa...")
roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512
)
print("[sentiment] Both models ready!")


def analyze_sentiment(text: str) -> dict:
    # VADER — great for short social text
    vader_scores = vader.polarity_scores(text)

    # RoBERTa — better for longer text
    roberta_result = roberta(text[:512])[0]

    # Clean up RoBERTa label (different model versions use different names)
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "NEGATIVE": "negative",
        "NEUTRAL":  "neutral",
        "POSITIVE": "positive",
    }
    clean_label = label_map.get(
        roberta_result["label"].upper(),
        roberta_result["label"].lower()
    )

    return {
        "vader": {
            "compound": round(vader_scores["compound"], 3),
            "positive": round(vader_scores["pos"], 3),
            "negative": round(vader_scores["neg"], 3),
            "neutral":  round(vader_scores["neu"], 3),
        },
        "roberta": {
            "label": clean_label,
            "confidence": round(roberta_result["score"], 3),
        },
        "summary": {
            "overall_tone": clean_label,
            "intensity": round(abs(vader_scores["compound"]), 3),
            "is_emotional": abs(vader_scores["compound"]) > 0.3,
        }
    }