FROM python:3.11-slim

# System dependencies needed by faiss and sentence-transformers
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY streamlit_app.py .

# Copy artifacts if they exist locally — otherwise mount at runtime (see docker-compose)
COPY lyric_search_artifacts/ ./lyric_search_artifacts/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
