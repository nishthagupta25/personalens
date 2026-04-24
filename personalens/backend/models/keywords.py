from keybert import KeyBERT

print("[keywords] Loading KeyBERT...")
kw_model = KeyBERT()
print("[keywords] KeyBERT ready!")


def extract_keywords(text: str, top_n: int = 10) -> dict:

    # Single words
    single_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        top_n=top_n
    )

    # Phrases (1-2 words, more meaningful)
    phrase_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        use_mmr=True,
        diversity=0.5,
        top_n=top_n
    )

    single_list = [
        {"word": kw, "score": round(score, 3)}
        for kw, score in single_keywords
    ]

    phrase_list = [
        {"phrase": kw, "score": round(score, 3)}
        for kw, score in phrase_keywords
    ]

    top_plain = [kw for kw, score in phrase_keywords[:5]]

    return {
        "single_keywords": single_list,
        "phrases": phrase_list,
        "top_5": top_plain
    }