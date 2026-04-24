from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline.orchestrator import run_pipeline
import time

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
        raise HTTPException(
            status_code=400,
            detail="Text too short. Please provide at least 30 characters."
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Please provide less than 5000 characters."
        )

    start = time.time()
    result = run_pipeline(text)
    elapsed = round(time.time() - start, 2)

    result["processing_time_seconds"] = elapsed
    result["source_type"] = body.source_type
    return result

@app.get("/models")
def models_status():
    return {
        "models": [
            {"name": "VADER", "status": "loaded", "type": "sentiment"},
            {"name": "RoBERTa", "status": "loaded", "type": "sentiment"},
            {"name": "KeyBERT", "status": "loaded", "type": "keywords"},
            {"name": "spaCy", "status": "loaded", "type": "linguistic"},
            {"name": "Emotion classifier", "status": "loaded", "type": "emotions"},
            {"name": "MiniLM", "status": "loaded", "type": "embeddings"},
            {"name": "Groq LLM", "status": "loaded", "type": "reasoning"},
        ]
    }