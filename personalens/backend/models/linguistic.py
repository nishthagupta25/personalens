import spacy

print("[linguistic] Loading spaCy...")
nlp = spacy.load("en_core_web_sm")
print("[linguistic] spaCy ready!")


def extract_features(text: str) -> dict:
    doc = nlp(text)

    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]

    # Count meaningful signals
    question_count = sum(1 for s in sentences if s.text.strip().endswith("?"))
    exclamation_count = text.count("!")
    first_person = sum(1 for t in doc if t.text.lower() in {
        "i", "me", "my", "myself", "mine"
    })
    hedge_words = sum(1 for t in doc if t.text.lower() in {
        "maybe", "perhaps", "might", "could", "probably",
        "possibly", "seems", "apparently", "somehow", "guess"
    })
    assertive_words = sum(1 for t in doc if t.text.lower() in {
        "always", "never", "definitely", "absolutely",
        "certainly", "clearly", "obviously", "must"
    })
    negative_words = sum(1 for t in doc if t.text.lower() in {
        "not", "no", "never", "nothing", "nobody", "neither"
    })

    total_words = max(len(words), 1)
    total_sents = max(len(sentences), 1)

    # Vocabulary richness (unique words / total words)
    unique_ratio = len(set(t.lemma_.lower() for t in words)) / total_words

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(len(words) / total_sents, 1),
        "question_ratio": round(question_count / total_sents, 2),
        "exclamation_ratio": round(exclamation_count / total_sents, 2),
        "first_person_ratio": round(first_person / total_words, 2),
        "hedge_word_ratio": round(hedge_words / total_words, 2),
        "assertive_ratio": round(assertive_words / total_words, 2),
        "negative_ratio": round(negative_words / total_words, 2),
        "vocabulary_richness": round(unique_ratio, 2),
        "signals": {
            "is_questioning": question_count > 1,
            "is_assertive": assertive_words > hedge_words,
            "is_self_focused": round(first_person / total_words, 2) > 0.05,
            "is_expressive": exclamation_count > 1,
        }
    }