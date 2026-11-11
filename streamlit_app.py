import os
os.environ.setdefault("HF_HOME", "/tmp/hf")

import numpy as np
import streamlit as st
import faiss, PyPDF2, docx
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="RAG App with Groq - Saman Azeemi",
    page_icon="📚",
    layout="centered",
)

st.markdown("""
<style>
.hero {
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #064e3b 0%, #047857 55%, #c9a227 100%);
    box-shadow: 0 10px 30px rgba(4, 120, 87, 0.35);
    border: 1px solid rgba(201, 162, 39, 0.6);
}
.hero h1 { color: #ffffff; font-size: 1.75rem; line-height: 1.3; margin: 0; padding: 0; }
.hero p  { color: rgba(255,255,255,0.88); margin: 0.5rem 0 0 0; font-size: 0.95rem; }
.step {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
    background: rgba(201, 162, 39, 0.18); color: #c9a227;
    font-weight: 700; font-size: 0.8rem; letter-spacing: 0.04em; margin-bottom: 0.4rem;
}
.stButton > button {
    border-radius: 999px; font-weight: 600; padding: 0.5rem 1.4rem;
    color: #ffffff; background: linear-gradient(135deg, #065f46, #059669);
    border: 1px solid #c9a227;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    color: #ffffff; transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(201, 162, 39, 0.5);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📚 Open-Source RAG App with Groq - Saman Azeemi (B04-0923-000053)</h1>
  <p>CS4106 — Foundations of Generative AI · Answers grounded in your own documents</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = ""
    if api_key:
        st.success("🔑 Groq key loaded from app secret")
    else:
        st.error("🔑 GROQ_API_KEY secret is missing")
    model = st.selectbox(
        "🧠 Model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    )
    top_k = st.slider("📑 Chunks to retrieve (k)", 1, 5, 3)
    st.divider()
    st.caption("How it works")
    st.markdown("""
1. **Chunk** the document
2. **Embed** with MiniLM
3. **Index** in FAISS
4. **Retrieve** top-k
5. **Generate** with Groq
""")
    st.divider()
    if st.button("🗑️ Clear knowledge base"):
        st.session_state.pop("chunks", None)
        st.session_state.pop("index", None)
        st.session_state.pop("n_files", None)
        st.toast("Knowledge base cleared")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


def read_file(f):
    if f.name.endswith(".pdf"):
        pages = PyPDF2.PdfReader(f).pages
        return "".join(p.extract_text() or "" for p in pages)
    if f.name.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(f).paragraphs)
    return f.read().decode("utf-8", "ignore")


def chunk_text(text, size=150, overlap=30):
    words = text.split()
    out = []
    for i in range(0, len(words), size - overlap):
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return out


def build_index(chunks):
    emb = embedder.encode(chunks)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype("float32"))
    return idx


def search(idx, chunks, query, k):
    q = embedder.encode([query])
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    scores, ids = idx.search(q.astype("float32"), k)
    out = []
    for s, i in zip(scores[0], ids[0]):
        if 0 <= i < len(chunks):
            out.append((chunks[i], float(s)))
    return out


# ---- Step 1: upload ----
st.markdown('<span class="step">STEP 1 · UPLOAD</span>', unsafe_allow_html=True)
files = st.file_uploader(
    "Upload PDF / DOCX / TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

# ---- Step 2: build ----
st.markdown('<span class="step">STEP 2 · BUILD</span>', unsafe_allow_html=True)
if st.button("⚡ Build knowledge base"):
    if not files:
        st.warning("Upload at least one file first.")
    else:
        with st.spinner("Reading and indexing..."):
            text = "\n".join(read_file(f) for f in files)
            chunks = chunk_text(text)
            if chunks:
                st.session_state.chunks = chunks
                st.session_state.index = build_index(chunks)
                st.session_state.n_files = len(files)
        if chunks:
            st.toast("Knowledge base ready", icon="✅")
        else:
            st.error("No text found. Scanned PDFs will not work.")

if "index" in st.session_state:
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Files", st.session_state.get("n_files", 0))
    c2.metric("🧩 Chunks", len(st.session_state.chunks))
    c3.metric("🔎 Retrieving", f"top {top_k}")
else:
    st.info("No knowledge base yet. Upload a document and click Build.")

# ---- Step 3: ask ----
st.markdown('<span class="step">STEP 3 · ASK</span>', unsafe_allow_html=True)
question = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g. How many days of annual leave?",
)
if st.button("✨ Answer") and question:
    if not api_key:
        st.error("Add GROQ_API_KEY in the app secrets.")
    elif "index" not in st.session_state:
        st.error("Upload a document and click Build first.")
    else:
        with st.spinner("Searching and asking Groq..."):
            hits = search(
                st.session_state.index,
                st.session_state.chunks,
                question,
                top_k,
            )
            context = "\n\n".join(d for d, _ in hits)
            prompt = (
                "Answer using ONLY the context. "
                "If it is not there, say you don't know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
            client = Groq(api_key=api_key)
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        st.subheader("💬 Answer")
        with st.container(border=True):
            st.markdown(r.choices[0].message.content)
        with st.expander("📚 Sources (retrieved chunks)", expanded=True):
            for i, (d, s) in enumerate(hits, 1):
                st.markdown(f"**Chunk {i}** · similarity score {s:.2f}")
                st.progress(min(max(s, 0.0), 1.0))
                st.write(d)
                if i < len(hits):
                    st.divider()

st.caption(
    "Built by Saman Azeemi (B04-0923-000053) · "
    "Groq · Sentence-Transformers · FAISS · Streamlit"
)
