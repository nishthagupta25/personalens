from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
import json
import re
from groq import Groq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Load env (local) — Railway uses env vars directly
load_dotenv()

# Load lightweight models only
print("[lite] Loading VADER...")
vader = SentimentIntensityAnalyzer()

print("[lite] Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

print("[lite] Loading Groq...")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("[lite] All models ready!")

app = FastAPI(title="PersonaLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    source_type: str = "general"


def get_sentiment(text):
    scores = vader.polarity_scores(text)
    return {
        "vader": {
            "compound": round(scores["compound"], 3),
            "positive": round(scores["pos"], 3),
            "negative": round(scores["neg"], 3),
            "neutral":  round(scores["neu"], 3),
        },
        "summary": {
            "overall_tone": "positive" if scores["compound"] > 0.05 else "negative" if scores["compound"] < -0.05 else "neutral",
            "intensity": round(abs(scores["compound"]), 3),
            "is_emotional": abs(scores["compound"]) > 0.3,
        }
    }


def get_linguistic(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]
    total_words = max(len(words), 1)
    total_sents = max(len(sentences), 1)

    question_count = sum(1 for s in sentences if s.text.strip().endswith("?"))
    exclamation_count = text.count("!")
    first_person = sum(1 for t in doc if t.text.lower() in {"i","me","my","myself"})
    hedge_words = sum(1 for t in doc if t.text.lower() in {"maybe","perhaps","might","could","probably","possibly","seems"})
    assertive = sum(1 for t in doc if t.text.lower() in {"always","never","definitely","absolutely","certainly","clearly"})
    unique_ratio = len(set(t.lemma_.lower() for t in words)) / total_words

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(len(words)/total_sents, 1),
        "question_ratio": round(question_count/total_sents, 2),
        "exclamation_ratio": round(exclamation_count/total_sents, 2),
        "first_person_ratio": round(first_person/total_words, 2),
        "hedge_word_ratio": round(hedge_words/total_words, 2),
        "assertive_ratio": round(assertive/total_words, 2),
        "vocabulary_richness": round(unique_ratio, 2),
        "signals": {
            "is_questioning": question_count > 1,
            "is_assertive": assertive > hedge_words,
            "is_self_focused": round(first_person/total_words, 2) > 0.05,
            "is_expressive": exclamation_count > 1,
        }
    }


SYSTEM_PROMPT = """You are an expert psycholinguist and personality analyst.
You analyze text signals to infer personality traits using the Big Five (OCEAN) model.
You always respond ONLY with valid JSON — no markdown, no explanation outside the JSON.
CRITICAL: Return only valid JSON. Check every bracket and brace is properly closed."""


def build_prompt(text, sentiment, linguistic):
    return f"""Analyze the following person's writing and NLP signals.

--- WRITING SAMPLE ---
{text[:1500]}

--- NLP SIGNALS ---
Sentiment: compound={sentiment['vader']['compound']}, tone={sentiment['summary']['overall_tone']}
Linguistic features:
  - Word count: {linguistic['word_count']}
  - Avg sentence length: {linguistic['avg_sentence_length']}
  - Question ratio: {linguistic['question_ratio']}
  - Hedge word ratio: {linguistic['hedge_word_ratio']}
  - First person ratio: {linguistic['first_person_ratio']}
  - Vocabulary richness: {linguistic['vocabulary_richness']}
  - Is assertive: {linguistic['signals']['is_assertive']}
  - Is expressive: {linguistic['signals']['is_expressive']}
  - Is self focused: {linguistic['signals']['is_self_focused']}

--- YOUR TASK ---
Return ONLY valid JSON. No extra text before or after.

{{
  "ocean": {{
    "openness":          {{"score": 0, "evidence": "1 sentence"}},
    "conscientiousness": {{"score": 0, "evidence": "1 sentence"}},
    "extraversion":      {{"score": 0, "evidence": "1 sentence"}},
    "agreeableness":     {{"score": 0, "evidence": "1 sentence"}},
    "neuroticism":       {{"score": 0, "evidence": "1 sentence"}}
  }},
  "communication_style": {{
    "primary_style": "analytical/expressive/driver/amiable",
    "tone": "formal/casual/passionate/reserved/aggressive/nurturing",
    "description": "2-3 sentences"
  }},
  "behavioral_insights": {{
    "thinking_style": "e.g. systems thinker",
    "decision_making": "e.g. data-driven",
    "stress_signals": "what writing reveals under pressure",
    "social_orientation": "introverted/extroverted/ambiverted with evidence"
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "career_fit": ["role 1", "role 2", "role 3"],
  "confidence_score": 0
}}"""


def repair_json(raw):
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    raw = raw[start:end]
    raw = re.sub(r'\]\s*\n(\s*}})', r'}\n\1', raw)
    return raw


def run_llm(text, sentiment, linguistic):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(text, sentiment, linguistic)}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        raw = response.choices[0].message.content.strip()
        raw = repair_json(raw)
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "PersonaLens API is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/analyze")
def analyze(body: TextInput):
    text = body.text.strip()
    if len(text) < 30:
        raise HTTPException(status_code=400, detail="Text too short.")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long.")

    sentiment  = get_sentiment(text)
    linguistic = get_linguistic(text)
    analysis   = run_llm(text, sentiment, linguistic)

    return {
        "status": "ok",
        "text_length": len(text),
        "signals": {
            "sentiment": sentiment,
            "linguistic": linguistic,
        },
        "analysis": analysis
    }