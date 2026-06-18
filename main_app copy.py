#Import
#(“Streamlit builds the interactive UI 
# for asking questions and showing answers.”
#“Pickle loads stored embeddings locally.”)
#(“Regex for cleaning input.”
#“NumPy + cosine similarity for dense retrieval scoring.”
#“OpenAI client is used for generation (and optionally query embeddings).”
import streamlit as st
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
#(“SentenceTransformer is a fast bi-encoder for semantic comparisons.”
#“CrossEncoder is used for accurate reranking of retrieved candidates.”)
from sentence_transformers import SentenceTransformer, CrossEncoder
#(“BM25 provides lexical retrieval to complement dense retrieval.”)
from rank_bm25 import BM25Okapi

# Initialize models
#(“This follows a standard high-quality RAG pattern: 
# bi-encoder for speed, cross-encoder for precision.”
#“MiniLM is lightweight and fast for UI latency.”)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Text preprocessing
#(“Cleans user queries to remove noise and normalize whitespace.”
(#“Improves consistent tokenization and matching for both BM25 and embeddings.”)
def preprocess_text(text):
    text = re.sub(r'[^\w\s.,;:!?\-()\[\]\'"]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

#Prompt formatting (citation-aware grounding)
#(“Builds a structured context prompt from retrieved evidence.”)
def format_prompt(query, retrieved_docs):
#(“Each chunk is tagged with a source label so citations are enforceable.”
#“I truncate to control token usage and keep responses fast.”)
    context_blocks = []
    for doc in retrieved_docs:
        file_name = doc["metadata"].get("file_name", "source")
        context_blocks.append(f"[[SOURCE: {file_name}]]\n{doc['text'][:1000]}")
    context = "\n\n".join(context_blocks)
#((“This explicitly reduces hallucinations by
# forcing the model to ground answers.”
#“Adds a safe fallback when retrieval 
# doesn’t contain the answer.”)
#(“This explicitly reduces hallucinations 
# by forcing the model to ground answers.”)
#“Adds a safe fallback when retrieval 
# doesn’t contain the answer.”)
    prompt = f"""
You are a helpful assistant. Use the following sources to answer the question.
Always cite the source in the format [[SOURCE: name]]. 
If the answer is not found, say: "I don't have enough information."

## Sources:
{context}

## Question:
{query}

## Answer (with citations):
""".strip()
    return prompt

#Generate the answer (LLM call with grounding rules)
#(“I build a prompt that includes retrieved evidence 
# chunks and requires citations.”)

def generate_answer_openai(query, retrieved_docs, client, model_name="gpt-3.5-turbo"):
    prompt = format_prompt(query, retrieved_docs)
    #(“System message enforces citations globally.”
#“Low temperature improves consistency 
# and reduces hallucinations.”)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You must cite sources for all "
                "factual information using [[SOURCE: name]] format.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_citations(answer):
#(“Regex extracts all cited sources so I can display them in the UI.”)
    sources = set(re.findall(r"\[\[SOURCE: (.*?)\]\]", answer))
    if sources:
    #“Return the final grounded answer text.”
        return f"Sources: {', '.join(f'[{src}]' for src in sources)}"
        #(“This gives a clean citation summary 
        # and handles the no-citation case.”)

    return "Sources: None"

#(“This retrieval function uses query embeddings and
#  compares them to locally stored chunk embeddings.”)
def hybrid_search(
    query, local_embedded_docs, client, top_k=5, similarity_threshold=0.5
):
    # Get query embedding from OpenAI
    #(“I embed the query using the same embedding model 
    # used during indexing to stay in the same vector space.”)
    response = client.embeddings.create(input=[query], model="text-embedding-3-small")
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

    # Calculate cosine similarities with local documents
    doc_embeddings = np.array(
        [np.array(doc["embedding"]) for doc in local_embedded_docs]
    )
    #(“Compute cosine similarity between 
    # the query vector and all stored vectors.”)

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
#(“I apply an early similarity threshold 
# to drop weak matches and reduce downstream reranking cost.”)
    filtered_docs = [
        {"text": doc["document"], "metadata": doc["metadata"], "similarity": sim}
        for doc, sim in zip(local_embedded_docs, similarities)
        if sim >= similarity_threshold
    ]

    if not filtered_docs:
        return []
#(“Safe fallback: if nothing matches, 
# retrieval returns empty and the app can respond accordingly.”)
#“I over-retrieve (top_k * 3) to keep recall 
# high before reranking/merging steps.”)
    top_vector_hits = sorted(
        filtered_docs, key=lambda x: x["similarity"], reverse=True
    )[: top_k * 3]

    # BM25 lexical retrieval
    #(“I build a BM25 index over the 
    # corpus to capture exact keyword matches.”
#“BM25 complements dense embeddings, 
# especially for entities/names/dates.”)

    corpus = [
        {"text": doc["document"], "metadata": doc["metadata"]}
        for doc in local_embedded_docs
    ]
    tokenized_corpus = [doc["text"].split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
#(“I score documents and over-retrieve (top_k*3) 
# to preserve recall before reranking.”)
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[: top_k * 3]
    bm25_hits = [
        {"text": corpus[i]["text"], "metadata": corpus[i]["metadata"]}
        for i in top_bm25_indices
    ]

    # Merge dense + BM25 candidates
    #(“I merge results from both 
    # retrievers and deduplicate candidates.”)
    combined_docs = {doc["text"]: doc for doc in top_vector_hits + bm25_hits}
    combined_list = list(combined_docs.values())
    #Semantic filtering (fast bi-encoder)
    #(“This is a cheap second-stage filter to 
    # (remove weak candidates before cross-encoder reranking.”
#“It reduces latency and cost by shrinking the reranking set.”)
    query_emb = sentence_model.encode([query])
    doc_embs = sentence_model.encode([doc["text"] for doc in combined_list])
    sims = cosine_similarity(query_emb, doc_embs)[0]

    semantically_filtered = [
        doc for doc, sim in zip(combined_list, sims) if sim > similarity_threshold
    ]

    if not semantically_filtered:
        return []
        #(“Safe fallback if nothing passes semantic filtering.”)
    #Cross-encoder reranking (highest precision)
    #(“Cross-encoder jointly encodes query and passage, 
    # giving more accurate relevance scoring.”
#“Final output is the Top-K best 
# chunks used for grounded generation.”)

    cross_inp = [(query, doc["text"]) for doc in semantically_filtered]
    scores = cross_encoder.predict(cross_inp)
    reranked = sorted(
        zip(scores, semantically_filtered), key=lambda x: x[0], reverse=True
    )

    return [doc for _, doc in reranked[:top_k]]


# ----------------- Streamlit UI -----------------

#(“Simple UI entry point for interacting with the local RAG system.”)
st.title("📚 Local Q&A Chatbot with Citations")
#(“User provides an API key securely and
#  uploads the precomputed embeddings file.”)
#“This makes the app portable: 
# no hardcoded keys and no fixed dataset dependency.”)
openai_key = st.text_input("🔐 Enter your OpenAI API Key:", type="password")
uploaded_file = st.file_uploader("📂 Upload your embedded `.pkl` file", type=["pkl"])

if openai_key and uploaded_file:
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_key)
#(“Initialize OpenAI client dynamically.”)
#(“Load embeddings from the uploaded file so retrieval works locally.”)
        local_embedded_docs = pickle.load(uploaded_file)
        st.success(f"✅ Loaded {len(local_embedded_docs)} embedded chunks.")
        #Query → retrieve → generate
        query = st.text_input("❓ Ask a question:")
        if query:
            query = preprocess_text(query)
            with st.spinner("🔍 Searching and generating answer..."):
                #(“Runs retrieval with a loading spinner for good UX.”)
                retrieved_docs = hybrid_search(query, local_embedded_docs, client)

            if retrieved_docs:
                #(“Generates an answer with enforced citations,
                #  then displays answer + sources.”)
                answer = generate_answer_openai(query, retrieved_docs, client)
                st.markdown("### 🤖 Answer:")
                st.markdown(answer)
                st.markdown(f"#### 📚 {extract_citations(answer)}")
#(“Shows the top evidence chunks so the user can verify grounding.”
#“This increases trust and makes debugging easy.”)
                with st.expander("🔎 Top Supporting Documents"):
                    for i, doc in enumerate(retrieved_docs[:3]):
                        source = doc["metadata"].get("file_name", "unknown")
                        st.markdown(f"**{i+1}. Source: {source}**")
                        st.text(doc["text"][:500] + "...")
            else:
                st.warning("⚠️ No relevant documents found.")
                #(“Prevents the app from crashing and provides a useful error message.”)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
elif not openai_key:
    st.info("ℹ️ Please enter your OpenAI API key.")
