"""
🎵 Lyric Search — Streamlit App
Matches the `complete_pipeline.ipynb` backend: BM25 (sparse) + SBERT/FAISS (dense).

Usage:
    streamlit run streamlit_app.py

Required files (produced by the Colab notebook):
    songs_lyrics_metadata.csv   ← original corpus + metadata (title, artist, tag, lyrics_clean)
    songs.index                 ← FAISS index built from SBERT embeddings

Set the paths below or via environment variables / sidebar file uploader.
"""

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lyric Search",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.stApp { background: #0d0d14; color: #e8e4f0; }

section[data-testid="stSidebar"] {
    background: #13121e !important;
    border-right: 1px solid #2a2740;
}

.result-card {
    background: linear-gradient(135deg, #1a1828 0%, #1e1c2e 100%);
    border: 1px solid #2e2b45;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #7c6af7; }

.rank-badge {
    display: inline-block;
    background: #7c6af7;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}

.song-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f0ecff;
    margin: 0 0 2px;
}
.song-artist { font-size: 0.88rem; color: #9b93c9; margin-bottom: 0.6rem; }

.tag-pill {
    display: inline-block;
    background: #1e2a3a;
    color: #7cc3f5;
    font-size: 0.7rem;
    padding: 2px 9px;
    border-radius: 12px;
    border: 1px solid #2d4a5e;
    margin-right: 6px;
}

.score-pill {
    display: inline-block;
    background: #252240;
    color: #a098d8;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 12px;
    font-family: 'DM Mono', monospace;
    border: 1px solid #2e2b45;
}

.model-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 6px;
}
.badge-bm25   { background: #3a2810; color: #f5a623; border: 1px solid #5e4218; }
.badge-sbert  { background: #2a1e3a; color: #c47cf5; border: 1px solid #4a2e65; }

.lyric-preview {
    font-size: 0.82rem;
    color: #7a748f;
    font-style: italic;
    line-height: 1.55;
    white-space: pre-wrap;
    margin-top: 0.5rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c5b8f8 20%, #8b7de8 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

div[data-testid="stTextInput"] input {
    background: #1a1828 !important;
    border: 1.5px solid #2e2b45 !important;
    color: #e8e4f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 3px rgba(124,106,247,0.15) !important;
}

.stButton button {
    background: linear-gradient(135deg, #7c6af7, #9b55e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.8rem !important;
}
.stButton button:hover { opacity: 0.88; }

.stat-box {
    background: #1a1828;
    border: 1px solid #2e2b45;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.stat-num { font-size: 1.5rem; font-weight: 700; color: #c5b8f8; }
.stat-label { font-size: 0.72rem; color: #6b657a; text-transform: uppercase; letter-spacing: 0.08em; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
# These map to the notebook's DRIVE_BASE / CSV_PATH / INDEX_PATH.
# Override via environment variables, or drop the files next to this script.
DEFAULT_CSV_PATH   = os.environ.get("LYRICS_CSV_PATH", "songs_lyrics_metadata.csv")
DEFAULT_INDEX_PATH = os.environ.get("LYRICS_INDEX_PATH", "songs.index")
SBERT_MODEL_NAME    = os.environ.get("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
# When running in Docker, LYRICS_CSV_PATH / LYRICS_INDEX_PATH are set to
# /app/data/... via the Dockerfile and fed by the docker-compose volume mount.

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lyric_preview(text: str, max_chars: int = 220) -> str:
    if not isinstance(text, str):
        return ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    preview = " ".join(lines[:5]) if len(lines) <= 1 else "\n".join(lines[:5])
    if len(preview) > max_chars:
        preview = preview[:max_chars].rsplit(" ", 1)[0] + "…"
    return preview


# ── Load resources (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading corpus, BM25 index, SBERT model, and FAISS index…")
def load_resources(csv_path: str, index_path: str, model_name: str):
    if not Path(csv_path).exists():
        return None
    if not Path(index_path).exists():
        return None

    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(str)

    # Match the notebook's column fallback logic
    lyrics_col = "lyrics_clean" if "lyrics_clean" in df.columns else df.columns[-1]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    index = faiss.read_index(index_path)

    corpus = df[lyrics_col].fillna("").tolist()
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    return {
        "df": df,
        "model": model,
        "index": index,
        "bm25": bm25,
        "lyrics_col": lyrics_col,
    }


# ── Search functions (mirror notebook's search_sbert / search_bm25) ─────────
def search_sbert(query_text: str, res: dict, k: int) -> list[dict]:
    embedding = res["model"].encode([query_text], normalize_embeddings=True)
    distances, indices = res["index"].search(embedding, k)
    df = res["df"]
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[int(idx)]
        results.append({
            "rank": rank,
            "title": row.get("title", "Unknown"),
            "artist": row.get("artist", "Unknown"),
            "tag": row.get("tag", ""),
            "score": float(score),
            "lyrics": row.get(res["lyrics_col"], ""),
        })
    return results


def search_bm25(query_text: str, res: dict, k: int) -> list[dict]:
    tokens = query_text.lower().split()
    scores = res["bm25"].get_scores(tokens)
    top_indices = scores.argsort()[::-1][:k]
    df = res["df"]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[int(idx)]
        results.append({
            "rank": rank,
            "title": row.get("title", "Unknown"),
            "artist": row.get("artist", "Unknown"),
            "tag": row.get("tag", ""),
            "score": float(scores[idx]),
            "lyrics": row.get(res["lyrics_col"], ""),
        })
    return results


MODEL_META = {
    "SBERT + FAISS": {
        "fn": search_sbert,
        "badge": "badge-sbert",
        "label": "SBERT",
        "desc": "Dense retrieval via Sentence-BERT embeddings, ranked with FAISS cosine similarity.",
    },
    "BM25": {
        "fn": search_bm25,
        "badge": "badge-bm25",
        "label": "BM25",
        "desc": "Sparse retrieval baseline — lexical term matching with length normalisation.",
    },
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Search Settings")

    with st.expander("📂 Data paths", expanded=False):
        csv_path = st.text_input("Corpus CSV path", value=DEFAULT_CSV_PATH)
        index_path = st.text_input("FAISS index path", value=DEFAULT_INDEX_PATH)
        st.caption("These match `CSV_PATH` and `INDEX_PATH` from the notebook config.")

    chosen_model = st.radio(
        "Retrieval Model",
        list(MODEL_META.keys()),
        index=0,
        help="Choose the retrieval algorithm.",
    )
    top_k = st.slider("Results to return", min_value=3, max_value=20, value=10)
    show_lyrics = st.toggle("Show lyric preview", value=True)
    st.divider()
    st.markdown(
        f"**{MODEL_META[chosen_model]['label']}** — {MODEL_META[chosen_model]['desc']}"
    )
    st.divider()
    st.markdown("##### How it works")
    st.markdown(
        "1. Run `complete_pipeline.ipynb` in Colab on your corpus.\n"
        "2. Make sure `songs_lyrics_metadata.csv` and `songs.index` are accessible "
        "(same folder as this app, or set custom paths above).\n"
        "3. Enter a lyric fragment or a descriptive/emotional query to search."
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎵 Lyric Search</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#7a748f;font-size:1rem;margin-top:0.3rem">Search songs by lyric fragments, themes, or emotions.</p>',
    unsafe_allow_html=True,
)
st.markdown("")

resources = load_resources(csv_path, index_path, SBERT_MODEL_NAME)

if resources is None:
    st.warning(
        f"**Required files not found.**\n\n"
        f"- Looking for corpus CSV at: `{csv_path}`\n"
        f"- Looking for FAISS index at: `{index_path}`\n\n"
        "Run `complete_pipeline.ipynb` first, then place `songs_lyrics_metadata.csv` "
        "and `songs.index` next to this app (or update the paths in the sidebar).",
        icon="⚠️",
    )
    st.stop()

# Stats bar
df = resources["df"]
n_artists = df["artist"].nunique() if "artist" in df.columns else "—"
n_tags = df["tag"].nunique() if "tag" in df.columns else "—"
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, f"{len(df):,}", "Songs in corpus"),
    (c2, f"{n_artists:,}" if isinstance(n_artists, int) else n_artists, "Unique artists"),
    (c3, f"{n_tags}" if n_tags == "—" else f"{n_tags:,}", "Genres / tags"),
    (c4, "2", "Retrieval models"),
]:
    col.markdown(
        f'<div class="stat-box"><div class="stat-num">{num}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Search bar
qcol, bcol = st.columns([6, 1])
with qcol:
    query = st.text_input(
        "Search query",
        placeholder="e.g. 'feeling numb after a long period of emotional pain'",
        label_visibility="collapsed",
    )
with bcol:
    search_btn = st.button("Search", use_container_width=True)

# Results
if (search_btn or query) and query.strip():
    model_info = MODEL_META[chosen_model]
    with st.spinner(f"Searching with {chosen_model}…"):
        results = model_info["fn"](query.strip(), resources, top_k)

    badge_html = f'<span class="model-badge {model_info["badge"]}">{model_info["label"]}</span>'
    st.markdown(
        f"<p style='color:#6b657a;font-size:0.85rem;margin-bottom:1rem'>"
        f"Top <b>{len(results)}</b> results for <b>\"{query}\"</b> {badge_html}</p>",
        unsafe_allow_html=True,
    )

    if not results:
        st.info("No results found. Try a different query.")
    else:
        for r in results:
            preview = lyric_preview(r["lyrics"]) if show_lyrics else ""
            preview_block = f'<div class="lyric-preview">{preview}</div>' if preview else ""
            tag_block = f'<span class="tag-pill">{r["tag"]}</span>' if r.get("tag") else ""
            st.markdown(
                f"""
                <div class="result-card">
                    <span class="rank-badge">#{r['rank']}</span>
                    <div class="song-title">{r['title']}</div>
                    <div class="song-artist">{r['artist']}</div>
                    {tag_block}
                    <span class="score-pill">score {r['score']:.4f}</span>
                    {preview_block}
                </div>
                """,
                unsafe_allow_html=True,
            )

elif not query.strip() and search_btn:
    st.warning("Please enter a search query.", icon="🔍")
else:
    st.markdown(
        '<p style="color:#3d3852;text-align:center;margin-top:4rem;font-size:0.9rem">'
        "Type a lyric, theme, or feeling to begin searching…</p>",
        unsafe_allow_html=True,
    )
