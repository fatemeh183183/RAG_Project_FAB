📚 Local Q&A Chatbot with Citations (RAG System)
An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to ask questions over a local document corpus and receive citation-grounded answers using hybrid retrieval, reranking, and large language models.
This project demonstrates production-style RAG design, including chunking optimization, hybrid search, reranking, hallucination control, evaluation metrics, and a Streamlit UI.

🚀 Key Features


🔎 Hybrid Retrieval


Dense vector search (OpenAI embeddings)


Lexical search (BM25)




🧠 Multi-stage Reranking


Semantic filtering (bi-encoder)


Precision reranking (cross-encoder)




📎 Citation-Grounded Answers


Inline source attribution ([[SOURCE: file.txt]])


Transparent evidence display




🧪 Evaluation & Hallucination Detection


RAGAS (faithfulness, answer relevancy)


Semantic hallucination checks


ROUGE & BLEU metrics




💾 Offline Embedding Storage


Embeddings saved locally (.pkl)


No re-embedding required at runtime




🖥️ Interactive Streamlit UI


Upload embeddings


Ask questions


View answers, sources, and supporting documents





🏗️ Architecture Overview
Documents (.txt)
      ↓
Text Cleaning & Chunking (Optimal Chunk Size)
      ↓
OpenAI Embeddings
      ↓
Local Storage (.pkl)
      ↓
Hybrid Retrieval
   ├── Dense Vector Search
   ├── BM25 Lexical Search
      ↓
Candidate Merge & Deduplication
      ↓
Semantic Filtering (Bi-Encoder)
      ↓
Cross-Encoder Reranking
      ↓
LLM Answer Generation (with citations)
      ↓
Evaluation & UI Display


🧩 Tech Stack


Language: Python 3.11


LLM: OpenAI GPT-3.5


Embeddings: OpenAI text-embedding-3-small


Retrieval:


SentenceTransformers


BM25 (rank-bm25)




Reranking: Cross-Encoder (ms-marco-MiniLM-L-6-v2)


Evaluation:


RAGAS


ROUGE / BLEU




UI: Streamlit


Storage: Pickle (.pkl)



📂 Project Structure
New_FAB_Project/
│
├── main_app.py              # Streamlit RAG application
├── Chatbot_GPT.ipynb        # RAG pipeline & evaluation notebook
├── Chatbot_Open.ipynb       # Experiments & analysis
├── embedded_chunks_safe.pkl # Saved embeddings
├── ancient_greece_data/     # Source documents
├── .env                     # Environment variables
└── README.md


⚙️ Setup Instructions
1️⃣ Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

2️⃣ Install dependencies
pip install -U pip
pip install streamlit openai sentence-transformers rank-bm25 scikit-learn numpy

3️⃣ Add OpenAI API key
Create a .env file:
OPENAI_API_KEY=your_api_key_here


▶️ Run the Application
streamlit run main_app.py

Then open:
http://localhost:8501


🖱️ How to Use


Enter your OpenAI API key


Upload the saved embeddings file (.pkl)


Ask a question


View:


Answer with inline citations


Source list


Top supporting documents





📊 Evaluation & Quality Control
This project includes multiple layers of evaluation:


RAGAS Metrics


Faithfulness


Answer Relevancy




Hallucination Detection


Sentence-level semantic similarity checks




Lexical & Semantic Metrics


ROUGE-1 / ROUGE-2 / ROUGE-L


BLEU score


# Local Q&A Chatbot with Citations

This is a Streamlit-based RAG application for local document Q&A with citations.

## Live Demo
👉 [(https://ragprojectfab-jjcas8myfqnrgjgq25nymr.streamlit.app/)

## How to use
1. Open the app using the link above
2. Enter your own OpenAI API key
3. Upload the embedded `.pkl` file
4. Ask questions












