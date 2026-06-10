FROM python:3.10-slim

WORKDIR /app

COPY personalens/requirements_lite.txt .

RUN pip install --no-cache-dir -r requirements_lite.txt

RUN python -m spacy download en_core_web_sm

COPY personalens/ .

EXPOSE 8080

CMD ["/bin/sh", "-c", "cd /app/backend_lite && uvicorn main:app --host 0.0.0.0 --port 8080"]