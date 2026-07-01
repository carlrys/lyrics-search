FROM python:3.11-slim

# System dependencies needed by faiss and sentence-transformers
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY streamlit_app.py .

# Data files (corpus CSV + FAISS index) are NOT copied here —
# they're large/binary and expected to be mounted at runtime via docker-compose.
# See docker-compose.yml volumes section.

# Default paths inside the container — match the sidebar defaults in streamlit_app.py
ENV LYRICS_CSV_PATH=/app/data/songs_lyrics_metadata.csv
ENV LYRICS_INDEX_PATH=/app/data/songs.index
ENV SBERT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
