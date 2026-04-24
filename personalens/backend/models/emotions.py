from transformers import pipeline

print("[emotions] Loading emotion classifier...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    truncation=True,
    max_length=512
)
print("[emotions] Emotion model ready!")


def detect_emotions(text: str) -> dict:
    """
    Detects 7 emotions: joy, anger, sadness, fear,
    disgust, surprise, neutral

    Returns all scores + the dominant emotion
    """

    # Get scores for all 7 emotions
    results = emotion_classifier(text[:512])[0]

    # Sort by score descending
    sorted_emotions = sorted(results, key=lambda x: x["score"], reverse=True)

    # Clean into a simple dict
    emotion_scores = {
        item["label"].lower(): round(item["score"], 3)
        for item in sorted_emotions
    }

    # Top emotion
    dominant = sorted_emotions[0]["label"].lower()
    dominant_score = round(sorted_emotions[0]["score"], 3)

    # Top 3 emotions
    top_3 = [
        {"emotion": item["label"].lower(), "score": round(item["score"], 3)}
        for item in sorted_emotions[:3]
    ]

    return {
        "dominant_emotion": dominant,
        "dominant_score": dominant_score,
        "top_3": top_3,
        "all_scores": emotion_scores
    }