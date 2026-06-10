FROM python:3.10-slim

WORKDIR /app

COPY personalens/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY personalens/ .

EXPOSE 8000

CMD ["/bin/sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]