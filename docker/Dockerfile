## Use a lightweight Python base image
FROM python:3.11-slim

## Disable Python bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1

## Force unbuffered output for logs
ENV PYTHONUNBUFFERED=1

## Set default project root inside container
ENV DEDUPLICATION_ROOT=/app

## Set PYTHONPATH so `src.*` imports work everywhere
ENV PYTHONPATH=/app

## Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

## Set working directory
WORKDIR /app

## Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

## Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

## Copy project files
COPY . /app

## Default command: run FastAPI with uvicorn
ENTRYPOINT ["uvicorn", "src.core.service:app", "--host", "0.0.0.0", "--port", "8080"]
