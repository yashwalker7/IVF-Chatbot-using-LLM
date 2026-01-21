# Dockerfile for IVF Chatbot (Flask + static frontend)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# avoid building large caches
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && apt-get remove -y build-essential git \
 && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# copy app code
COPY . /app

# expose port used by Spaces (7860 is common; we will bind to it)
ENV PORT=7860

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "server:app", "--workers", "1", "--threads", "4", "--timeout", "120"]
