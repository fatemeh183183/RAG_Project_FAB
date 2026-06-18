# Imports
import os
import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------- Configuration -----------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_EMBEDDINGS_FILE = Path(os.getenv("EMBEDDINGS_FILE", "embedded_chunks_safe.pkl"))


# ----------------- Cached Resources -----------------
@st.cache_resource(show_spinner="Loading retrieval models...")
def load_retrieval_models():
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return sentence_model, cross_encoder


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


# ----------------- Text preprocessing -----------------
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^\w\s.,;:!?\-()\[\]\'"]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------- Prompt formatting -----------------
def format_prompt(query: str, retrieved_docs: list[dict]) -> str:
    context_blocks = []

    for doc in retrieved_docs:
        file_name = doc["metadata"].get("file_name", "source")
        context_blocks.append(f"[[SOURCE: {file_name}]]\n{doc['text'][:1000]}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant. Use ONLY the following sources to answer the question.
Always cite factual claims using the exact format [[SOURCE: name]].
If the answer is not found in the sources, say: "I don't have enough information."

## Sources:
{context}

## Question:
{query}

## Answer with citations:
""".strip()

    return prompt


# ----------------- Answer generation -----------------
def generate_answer_openai(
    query: str,
    retrieved_docs: list[dict],
    client: OpenAI,
    model_name: str = OPENAI_CHAT_MODEL,
) -> str:
    prompt = format_prompt(query, retrieved_docs)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a citation-grounded RAG assistant. "
                    "Use only the provided context and cite sources using [[SOURCE: name]]."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def extract_citations(answer: str) -> str:
    sources = set(re.findall(r"\[\[SOURCE: (.*?)\]\]", answer))
    if sources:
        return f"Sources: {', '.join(f'[{src}]' for src in sorted(sources))}"
    return "Sources: None"


# ----------------- Document loading -----------------
def load_embedded_docs(uploaded_file=None) -> list[dict]:
    if uploaded_file is not None:
        return pickle.load(uploaded_file)

    if DEFAULT_EMBEDDINGS_FILE.exists():
        with DEFAULT_EMBEDDINGS_FILE.open("rb") as file:
            return pickle.load(file)

    return []


# ----------------- Hybrid search -----------------
def hybrid_search(
    query: str,
    local_embedded_docs: list[dict],
    client: OpenAI,
    sentence_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
) -> list[dict]:
    # OpenAI embedding for query
    response = client.embeddings.create(input=[query], model=OPENAI_EMBEDDING_MODEL)
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

    # Dense vector search
    doc_embeddings = np.array([np.array(doc["embedding"]) for doc in local_embedded_docs])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    filtered_docs = [
        {"text": doc["document"], "metadata": doc["metadata"], "similarity": sim}
        for doc, sim in zip(local_embedded_docs, similarities)
        if sim >= similarity_threshold
    ]

    if not filtered_docs:
        return []

    top_vector_hits = sorted(filtered_docs, key=lambda x: x["similarity"], reverse=True)[
        : top_k * 3
    ]

    # BM25 lexical search
    corpus = [
        {"text": doc["document"], "metadata": doc["metadata"]}
        for doc in local_embedded_docs
    ]

    tokenized_corpus = [doc["text"].split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    top_bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[: top_k * 3]

    bm25_hits = [
        {"text": corpus[i]["text"], "metadata": corpus[i]["metadata"]}
        for i in top_bm25_indices
    ]

    # Merge dense + BM25 candidates
    combined_docs = {doc["text"]: doc for doc in top_vector_hits + bm25_hits}
    combined_list = list(combined_docs.values())

    if not combined_list:
        return []

    # Semantic filtering
    query_emb = sentence_model.encode([query])
    doc_embs = sentence_model.encode([doc["text"] for doc in combined_list])
    sims = cosine_similarity(query_emb, doc_embs)[0]

    semantically_filtered = [
        doc for doc, sim in zip(combined_list, sims) if sim > similarity_threshold
    ]

    if not semantically_filtered:
        return []

    # Cross-encoder reranking
    cross_inputs = [(query, doc["text"]) for doc in semantically_filtered]
    scores = cross_encoder.predict(cross_inputs)

    reranked = sorted(zip(scores, semantically_filtered), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in reranked[:top_k]]


# ----------------- Streamlit UI -----------------
st.set_page_config(
    page_title="RAG Q&A Chatbot",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Local Q&A Chatbot with Citations")
st.caption("Hybrid RAG system with OpenAI embeddings, BM25 retrieval and cross-encoder reranking.")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file or Azure Environment Variables.")
    st.stop()

client = get_openai_client(OPENAI_API_KEY)
sentence_model, cross_encoder = load_retrieval_models()

uploaded_file = st.file_uploader(
    "📂 Upload embedded `.pkl` file, or use the default local file",
    type=["pkl"],
)

try:
    local_embedded_docs = load_embedded_docs(uploaded_file)

    if not local_embedded_docs:
        st.warning(
            "No embedded documents found. Upload a `.pkl` file or include "
            "`embedded_chunks_safe.pkl` in the project folder."
        )
        st.stop()

    st.success(f"✅ Loaded {len(local_embedded_docs)} embedded chunks.")

    query = st.text_input("❓ Ask a question:")

    if query:
        query = preprocess_text(query)

        with st.spinner("🔍 Searching and generating answer..."):
            retrieved_docs = hybrid_search(
                query=query,
                local_embedded_docs=local_embedded_docs,
                client=client,
                sentence_model=sentence_model,
                cross_encoder=cross_encoder,
            )

            if retrieved_docs:
                answer = generate_answer_openai(query, retrieved_docs, client)

                st.markdown("### 🤖 Answer:")
                st.markdown(answer)
                st.markdown(f"#### 📚 {extract_citations(answer)}")

                with st.expander("🔎 Top Supporting Documents"):
                    for i, doc in enumerate(retrieved_docs[:3]):
                        source = doc["metadata"].get("file_name", "unknown")
                        st.markdown(f"**{i + 1}. Source: {source}**")
                        st.text(doc["text"][:500] + "...")
            else:
                st.warning("⚠️ No relevant documents found.")

except Exception as error:
    st.error(f"⚠️ Error: {error}")
