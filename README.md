# 🎵 Lyric Search App — Docker Usage Guide

This web app is was made for research and study purposes.

A semantic lyric search engine using TF-IDF, BM25, and Sentence-BERT + FAISS.
Built with Google Colab (backend) and Streamlit (frontend).

Run the app on any machine with Docker — no Python, no pip, no environment setup required.

---

## What You Need

| Requirement | Details |
|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Install and make sure it is running before you start |
| `songs_lyrics_metadata.csv` | Exported from your Colab notebook |
| `songs.index` | FAISS index exported from your Colab notebook |

> ⚠️ Docker Desktop must be **open and running** in the background before any of the commands below will work.

---

## First-Time Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/carlrys/lyrics-search.git
cd lyrics-search
```

### Step 2 — Create the data folder

```bash
mkdir data
```

### Step 3 — Add your data files

Copy `songs_lyrics_metadata.csv` and `songs.index` (both downloaded from Google Drive after running the Colab notebook) into the `data/` folder you just created.

Your folder should now look like this:

```
lyrics-search/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── streamlit_app.py
├── requirements.txt
└── data/
    ├── songs_lyrics_metadata.csv
    └── songs.index
```

### Step 4 — Build and run

```bash
docker compose up --build
```

This will:
1. Build the Docker image (downloads Python, installs all dependencies — takes **3–5 minutes** the first time)
2. Start the app container
3. Make the app available at **http://localhost:8501**

Open your browser and go to **http://localhost:8501**.

---

## Every Time After That

Once the image is built, you don't need `--build` anymore:

```bash
docker compose up
```

To stop the app:

```bash
docker compose down
```

Or press `Ctrl + C` in the terminal, then run `docker compose down`.

---

## Using the App

### Choosing a search model

Use the **sidebar on the left** to switch between retrieval models:

| Model | Best for |
|---|---|
| **SBERT + FAISS** | Descriptive or emotional queries — *"feeling lost and numb after heartbreak"* |
| **BM25** | Exact lyric fragments — *"we found love in a hopeless place"* |

### Adjusting results

- **Results to return** — drag the slider to show between 3 and 20 results
- **Show lyric preview** — toggle on/off to show a short excerpt from each matched song

### Reading the results

Each result card shows:
- **Rank** — position in the result list (1 = best match)
- **Song title and artist**
- **Genre/tag** — if present in your dataset
- **Score** — how confident the model is in the match (closer to 1.0 = stronger match)
- **Lyric preview** — first few lines of the matched song's lyrics

### Understanding the score

| Model | Score range | What it means |
|---|---|---|
| SBERT + FAISS | 0.0 – 1.0 | Cosine similarity between query and song embeddings |
| BM25 | 0.0 – varies | BM25 relevance score (not normalized — relative ranking matters more than the number itself) |

---

## If the App Shows a Warning Instead of Results

> **"Required files not found"**

This means the `data/` folder is missing or the files inside it are named incorrectly. Check:

1. The `data/` folder exists inside your repo root
2. The files are named exactly:
   - `songs_lyrics_metadata.csv`
   - `songs.index`
3. The container was restarted after adding the files:
   ```bash
   docker compose down
   docker compose up
   ```

---

## Rebuilding After Code Changes

If you pull new changes from GitHub or edit `streamlit_app.py` or `requirements.txt`:

```bash
docker compose down
docker compose up --build
```

The `--build` flag forces Docker to rebuild the image with the latest code.

---

## Stopping and Cleaning Up

**Stop the app:**
```bash
docker compose down
```

**Remove the built image** (frees disk space — you'll need to rebuild next time):
```bash
docker compose down --rmi all
```

**Remove everything including the data volume:**
```bash
docker compose down --rmi all --volumes
```
