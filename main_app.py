#Imports
import streamlit as st
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# Initialize models
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^\w\s.,;:!?\-()\[\]\'"]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

#Prompt formatting (citation-aware grounding)
def format_prompt(query, retrieved_docs):
    context_blocks = []
    for doc in retrieved_docs:
        file_name = doc["metadata"].get("file_name", "source")
        context_blocks.append(f"[[SOURCE: {file_name}]]\n{doc['text'][:1000]}")
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant. Use the following sources to answer the question.
Always cite the source in the format [[SOURCE: name]]. If the answer is not found, say: "I don't have enough information."

## Sources:
{context}

## Question:
{query}

## Answer (with citations):
""".strip()
    return prompt

#Generate the answer (LLM call with grounding rules)
def generate_answer_openai(query, retrieved_docs, client, model_name="gpt-3.5-turbo"):
    prompt = format_prompt(query, retrieved_docs)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You must cite sources for all factual information using [[SOURCE: name]] format.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_citations(answer):
    sources = set(re.findall(r"\[\[SOURCE: (.*?)\]\]", answer))
    if sources:
        return f"Sources: {', '.join(f'[{src}]' for src in sources)}"
    return "Sources: None"


def hybrid_search(
    query, local_embedded_docs, client, top_k=5, similarity_threshold=0.5
):
    # Get query embedding from OpenAI
    response = client.embeddings.create(input=[query], model="text-embedding-3-small")
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

    # Calculate cosine similarities with local documents
    doc_embeddings = np.array(
        [np.array(doc["embedding"]) for doc in local_embedded_docs]
    )
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    filtered_docs = [
        {"text": doc["document"], "metadata": doc["metadata"], "similarity": sim}
        for doc, sim in zip(local_embedded_docs, similarities)
        if sim >= similarity_threshold
    ]

    if not filtered_docs:
        return []

    top_vector_hits = sorted(
        filtered_docs, key=lambda x: x["similarity"], reverse=True
    )[: top_k * 3]

    # BM25 lexical retrieval
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
    #Semantic filtering (fast bi-encoder)
    query_emb = sentence_model.encode([query])
    doc_embs = sentence_model.encode([doc["text"] for doc in combined_list])
    sims = cosine_similarity(query_emb, doc_embs)[0]

    semantically_filtered = [
        doc for doc, sim in zip(combined_list, sims) if sim > similarity_threshold
    ]

    if not semantically_filtered:
        return []
    #Cross-encoder reranking (highest precision)
    cross_inp = [(query, doc["text"]) for doc in semantically_filtered]
    scores = cross_encoder.predict(cross_inp)
    reranked = sorted(
        zip(scores, semantically_filtered), key=lambda x: x[0], reverse=True
    )

    return [doc for _, doc in reranked[:top_k]]


# ----------------- Streamlit UI -----------------
#App title
st.title("📚 Local Q&A Chatbot with Citations")

openai_key = st.text_input("🔐 Enter your OpenAI API Key:", type="password")
uploaded_file = st.file_uploader("📂 Upload your embedded `.pkl` file", type=["pkl"])

if openai_key and uploaded_file:
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_key)

        local_embedded_docs = pickle.load(uploaded_file)
        st.success(f"✅ Loaded {len(local_embedded_docs)} embedded chunks.")
        #Query → retrieve → generate
        query = st.text_input("❓ Ask a question:")
        if query:
            query = preprocess_text(query)
            with st.spinner("🔍 Searching and generating answer..."):
                retrieved_docs = hybrid_search(query, local_embedded_docs, client)

            if retrieved_docs:
                answer = generate_answer_openai(query, retrieved_docs, client)
                st.markdown("### 🤖 Answer:")
                st.markdown(answer)
                st.markdown(f"#### 📚 {extract_citations(answer)}")

                with st.expander("🔎 Top Supporting Documents"):
                    for i, doc in enumerate(retrieved_docs[:3]):
                        source = doc["metadata"].get("file_name", "unknown")
                        st.markdown(f"**{i+1}. Source: {source}**")
                        st.text(doc["text"][:500] + "...")
            else:
                st.warning("⚠️ No relevant documents found.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
elif not openai_key:
    st.info("ℹ️ Please enter your OpenAI API key.")
