# Image officielle légère Python 3.11
FROM python:3.11-slim

# Évite l'écriture de bytecode et force l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source et des modèles
COPY src/ ./src/
COPY models/ ./models/
COPY credentials.json* ./

EXPOSE 8000
EXPOSE 8501

# Commande par défaut (FastAPI)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

