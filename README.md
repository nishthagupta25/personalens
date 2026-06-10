# PersonaLens 🔍
### AI-Powered Personality Analysis from Text

## 🚀 Live Demo
👉 [Try PersonaLens](https://nishthagupta25.github.io/personalens/personalens/frontend/index.html)

> Paste any text — tweets, chats, essays — and get a deep personality profile powered by a 3-layer NLP pipeline.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

---

## 🧠 What It Does

PersonaLens analyzes text and outputs:
- **Big Five (OCEAN) personality traits** with evidence
- **Communication style** — analytical, expressive, driver, amiable
- **Behavioral insights** — thinking style, decision making, stress signals
- **Strengths & blind spots**
- **Career fit suggestions**

---

## ⚙️ Architecture — 3-Layer NLP Pipeline

```
Raw Text Input
     │
     ▼
Layer 1 — Classical NLP (fast, free)
  ├── VADER Sentiment Analysis
  ├── RoBERTa Sentiment (cardiffnlp)
  ├── KeyBERT Keyword Extraction
  └── spaCy Linguistic Features
     │
     ▼
Layer 2 — Deep NLP (HuggingFace)
  ├── Emotion Detection (j-hartmann/emotion-english)
  └── Sentence Embeddings (all-MiniLM-L6-v2)
     │
     ▼
Layer 3 — LLM Reasoning (Groq)
  └── Llama 3.3 70B → OCEAN Profile + Insights
     │
     ▼
Structured JSON Response → Beautiful Frontend UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python |
| Sentiment | VADER + cardiffnlp/twitter-roberta |
| Keywords | KeyBERT |
| Linguistic | spaCy |
| Emotions | j-hartmann/emotion-english-distilroberta |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq API (Llama 3.3 70B) |
| Frontend | HTML + CSS + Vanilla JS |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/personalens.git
cd personalens
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set up environment variables
```bash
# Create .env file
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the backend
```bash
cd backend
uvicorn main:app --reload
```

### 6. Open the frontend
Open `frontend/index.html` in your browser.

---

## 📊 Sample Output

```json
{
  "ocean": {
    "openness": {"score": 8, "evidence": "..."},
    "conscientiousness": {"score": 7, "evidence": "..."},
    "extraversion": {"score": 2, "evidence": "..."},
    "agreeableness": {"score": 5, "evidence": "..."},
    "neuroticism": {"score": 1, "evidence": "..."}
  },
  "communication_style": {"primary_style": "analytical", "tone": "reserved"},
  "strengths": ["problem-solving", "technical expertise", "self-motivation"],
  "career_fit": ["software engineer", "data analyst", "researcher"]
}
```

---

## 👩‍💻 Author
**Nishtha Gupta**