import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

print("[llm] Loading Groq client...")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("[llm] Groq client ready!")

SYSTEM_PROMPT = """You are an expert psycholinguist and personality analyst.
You analyze text signals to infer personality traits using the Big Five (OCEAN) model.
You always respond ONLY with valid JSON — no markdown, no explanation outside the JSON.
Be specific and evidence-based. Reference actual signals from the data provided.
CRITICAL: Return only valid JSON. Check every bracket and brace is properly closed."""


def build_prompt(text: str, signals: dict) -> str:
    sentiment  = signals["sentiment"]
    keywords   = signals["keywords"]
    linguistic = signals["linguistic"]
    emotions   = signals["emotions"]

    return f"""Analyze the following person's writing and NLP signals.

--- WRITING SAMPLE ---
{text[:1500]}

--- NLP SIGNALS ---
Sentiment: compound={sentiment['vader']['compound']}, tone={sentiment['roberta']['label']}, confidence={sentiment['roberta']['confidence']}
Dominant emotion: {emotions['dominant_emotion']} (score={emotions['dominant_score']})
Top 3 emotions: {emotions['top_3']}
Linguistic features:
  - Word count: {linguistic['word_count']}
  - Avg sentence length: {linguistic['avg_sentence_length']} words
  - Question ratio: {linguistic['question_ratio']}
  - Hedge word ratio: {linguistic['hedge_word_ratio']}
  - First person ratio: {linguistic['first_person_ratio']}
  - Vocabulary richness: {linguistic['vocabulary_richness']}
  - Is assertive: {linguistic['signals']['is_assertive']}
  - Is expressive: {linguistic['signals']['is_expressive']}
  - Is self focused: {linguistic['signals']['is_self_focused']}
Top keywords: {keywords['top_5']}

--- YOUR TASK ---
Return ONLY valid JSON. No extra text before or after.
Every opening bracket must have a closing bracket.

{{
  "ocean": {{
    "openness":          {{"score": 0, "evidence": "1 sentence citing signals"}},
    "conscientiousness": {{"score": 0, "evidence": "1 sentence"}},
    "extraversion":      {{"score": 0, "evidence": "1 sentence"}},
    "agreeableness":     {{"score": 0, "evidence": "1 sentence"}},
    "neuroticism":       {{"score": 0, "evidence": "1 sentence"}}
  }},
  "communication_style": {{
    "primary_style": "one of: analytical / expressive / driver / amiable",
    "tone": "one of: formal / casual / passionate / reserved / aggressive / nurturing",
    "description": "2-3 sentences about how this person communicates"
  }},
  "behavioral_insights": {{
    "thinking_style": "e.g. systems thinker, intuitive, detail-oriented",
    "decision_making": "e.g. data-driven, emotional, impulsive",
    "stress_signals": "what the writing reveals under pressure",
    "social_orientation": "introverted / extroverted / ambiverted with evidence"
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "career_fit": ["role 1", "role 2", "role 3"],
  "confidence_score": 0
}}"""


def repair_json(raw: str) -> str:
    """
    Attempts to fix common LLM JSON mistakes:
    - ] instead of } in objects
    - Missing closing braces
    """
    # Remove markdown
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    raw = raw.strip()

    # Extract JSON object
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    raw   = raw[start:end]

    # Fix ] used instead of } inside objects
    # Pattern: "] } " or "]\n  }" at end of ocean block
    raw = re.sub(r'\]\s*\n(\s*}})', r'}\n\1', raw)

    return raw


def run_llm_analysis(text: str, signals: dict) -> dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(text, signals)}
            ],
            temperature=0.2,
            max_tokens=1500
        )

        raw = response.choices[0].message.content.strip()
        raw = repair_json(raw)

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[llm] JSON parse error: {e}")
        print(f"[llm] Raw output: {raw}")
        return {"error": "LLM returned invalid JSON", "raw": raw}
    except Exception as e:
        print(f"[llm] Error: {e}")
        return {"error": str(e)}