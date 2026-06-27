"""
🎵 Lyric Search — Streamlit App
Supports TF-IDF, BM25, and Sentence-BERT + FAISS retrieval.

Usage:
    streamlit run streamlit_app.py

Artifacts required (produced by the Colab notebook):
    lyric_search_artifacts/
        tfidf_vectorizer.pkl
        tfidf_matrix.pkl
        bm25.pkl
        faiss.index
        embeddings.npy
        metadata.parquet
        config.json
"""

import re
import json
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

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
.badge-tfidf  { background: #1e3a3a; color: #5ecfb3; border: 1px solid #2d5e55; }
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

# ── Helpers ───────────────────────────────────────────────────────────────────
ARTIFACTS = Path("lyric_search_artifacts")

def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lyric_preview(text: str, max_chars: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    # Show first few meaningful lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    preview = "\n".join(lines[:5])
    return textwrap.shorten(preview, width=max_chars, placeholder="…")

# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval models…")
def load_artifacts():
    if not ARTIFACTS.exists():
        return None

    with open(ARTIFACTS / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)
    with open(ARTIFACTS / "tfidf_matrix.pkl", "rb") as f:
        tfidf_mat = pickle.load(f)
    with open(ARTIFACTS / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    faiss_idx = faiss.read_index(str(ARTIFACTS / "faiss.index"))
    embeddings = np.load(ARTIFACTS / "embeddings.npy")
    metadata = pd.read_parquet(ARTIFACTS / "metadata.parquet")

    with open(ARTIFACTS / "config.json") as f:
        config = json.load(f)

    # Load SBERT lazily (requires sentence-transformers)
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(config["sbert_model"])
    except Exception:
        sbert = None

    return {
        "tfidf_vec": tfidf_vec,
        "tfidf_mat": tfidf_mat,
        "bm25": bm25,
        "faiss_idx": faiss_idx,
        "embeddings": embeddings,
        "metadata": metadata,
        "config": config,
        "sbert": sbert,
    }


def tfidf_search(query: str, arts, top_k: int) -> list[dict]:
    q_vec = arts["tfidf_vec"].transform([clean_lyrics(query)])
    scores = cosine_similarity(q_vec, arts["tfidf_mat"]).flatten()
    idx = np.argsort(scores)[::-1][:top_k]
    meta = arts["metadata"]
    return [
        {
            "rank": i + 1,
            "title": meta.iloc[j]["title"],
            "artist": meta.iloc[j]["artist"],
            "score": float(scores[j]),
            "lyrics": meta.iloc[j]["lyrics"],
        }
        for i, j in enumerate(idx)
    ]


def bm25_search(query: str, arts, top_k: int) -> list[dict]:
    tokens = clean_lyrics(query).split()
    scores = np.array(arts["bm25"].get_scores(tokens))
    idx = np.argsort(scores)[::-1][:top_k]
    meta = arts["metadata"]
    return [
        {
            "rank": i + 1,
            "title": meta.iloc[j]["title"],
            "artist": meta.iloc[j]["artist"],
            "score": float(scores[j]),
            "lyrics": meta.iloc[j]["lyrics"],
        }
        for i, j in enumerate(idx)
    ]


def dense_search(query: str, arts, top_k: int) -> list[dict]:
    if arts["sbert"] is None:
        st.error("sentence-transformers not installed in this environment.")
        return []
    q_emb = arts["sbert"].encode(
        [clean_lyrics(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    scores, indices = arts["faiss_idx"].search(q_emb, top_k)
    meta = arts["metadata"]
    return [
        {
            "rank": i + 1,
            "title": meta.iloc[j]["title"],
            "artist": meta.iloc[j]["artist"],
            "score": float(s),
            "lyrics": meta.iloc[j]["lyrics"],
        }
        for i, (j, s) in enumerate(zip(indices[0], scores[0]))
    ]


MODEL_META = {
    "TF-IDF": {
        "fn": tfidf_search,
        "badge": "badge-tfidf",
        "label": "TF-IDF",
        "desc": "Term frequency-inverse document frequency. Fast, lexical matching.",
    },
    "BM25": {
        "fn": bm25_search,
        "badge": "badge-bm25",
        "label": "BM25",
        "desc": "Okapi BM25 — improved sparse ranking with document length normalisation.",
    },
    "SBERT + FAISS": {
        "fn": dense_search,
        "badge": "badge-sbert",
        "label": "SBERT",
        "desc": "Sentence-BERT embeddings retrieved via FAISS. Understands semantics.",
    },
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Search Settings")
    chosen_model = st.radio(
        "Retrieval Model",
        list(MODEL_META.keys()),
        index=2,
        help="Choose the retrieval algorithm.",
    )
    top_k = st.slider("Results to return", min_value=3, max_value=20, value=8)
    show_lyrics = st.toggle("Show lyric preview", value=True)
    st.divider()
    st.markdown(
        f"**{MODEL_META[chosen_model]['label']}** — {MODEL_META[chosen_model]['desc']}"
    )
    st.divider()
    st.markdown("##### How it works")
    st.markdown(
        "1. Run the Colab notebook on your CSV to build retrieval artifacts.\n"
        "2. Place the `lyric_search_artifacts/` folder next to this file.\n"
        "3. Enter a lyric fragment or semantic description to search."
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎵 Lyric Search</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#7a748f;font-size:1rem;margin-top:0.3rem">Search songs by lyric fragments or describe what you\'re looking for.</p>',
    unsafe_allow_html=True,
)
st.markdown("")

arts = load_artifacts()

if arts is None:
    st.warning(
        "**Artifacts not found.** Run the Colab notebook first, then place the "
        "`lyric_search_artifacts/` folder in the same directory as this app.",
        icon="⚠️",
    )
    st.stop()

# Stats bar
cfg = arts["config"]
meta = arts["metadata"]
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, f"{cfg['corpus_size']:,}", "Songs in corpus"),
    (c2, cfg["sbert_model"], "SBERT model"),
    (c3, f"{cfg['embedding_dim']}d", "Embedding dim"),
    (c4, "3", "Retrieval models"),
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
        placeholder="e.g.  'heart of gold'  or  'feeling lost on the highway'",
        label_visibility="collapsed",
    )
with bcol:
    search_btn = st.button("Search", use_container_width=True)

# Results
if (search_btn or query) and query.strip():
    model_info = MODEL_META[chosen_model]
    with st.spinner(f"Searching with {chosen_model}…"):
        results = model_info["fn"](query.strip(), arts, top_k)

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
            preview_block = (
                f'<div class="lyric-preview">{preview}</div>' if preview else ""
            )
            st.markdown(
                f"""
                <div class="result-card">
                    <span class="rank-badge">#{r['rank']}</span>
                    <div class="song-title">{r['title']}</div>
                    <div class="song-artist">{r['artist']}</div>
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
        "Type a lyric or describe a feeling to begin searching…</p>",
        unsafe_allow_html=True,
    )
