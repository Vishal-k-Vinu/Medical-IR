   Medical Information Retrieval System

A domain-specific semantic search engine for retrieving medical information using vector embeddings, Pinecone, SentenceTransformers, and Groq LLM query enhancement.

    Overview

* This task implements a full Medical Information Retrieval (IR) System capable of:
* Generating vector embeddings from medical text
* Storing and retrieving vectors using Pinecone Vector Database
* Enhancing user queries using a Groq LLM 
* Returning top-ranked results with similarity scores
* Displaying them in a simple Streamlit UI
* The system handles Medium-large datasets (up to 50000 rows) generated using gpt consisting of:
Diseases
Symptoms
Causes
Remedies & treatments
Prevention
Risk factors

      Project Architecture

medical-ir-system/
│
├── data/
│   └── medical_dataset.csv
│
├── src/
│   ├── embeddings/
│   │   └── embed_upsert.py
│   │
│   ├── searching/
│   │   └── search_engine.py
│   │
│   ├── query_enhance/
│   │   └── query_intelligence.py
│   │
│   └── ui/
│       └── app.py
│
├
└── README.md



     Workflow Diagram

                ┌──────────────────────────┐
                │        User Input        │
                │  (medical query typed)   │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  Query Enhancement (LLM) │
                │  Groq  rewrites          │
                │  the query for clarity   │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   Embedding Generation   │
                │ SentenceTransformer      │
                │ creates vector embedding │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  Pinecone Vector Search  │
                │ Computes similarity      │
                │ ranks top-K results      │
                │ │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │    Ranked Search Output  │
                │  Top results returned    │
                │  with metadata + scores  │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │    Streamlit UI Display  │
                │ Shows enhanced query,    │
                │ results & similarity     │
                │ scores in interface      │
                └──────────────────────────┘



      Tech Stack

* Embeddings	SentenceTransformers (MiniLM-L6-v2)
* Vector DB	Pinecone (Serverless, cosine similarity)
* Query Intelligence using Groq LLM 
* UI	Streamlit
* Language	Python
* Data Format	CSV

      Dataset Structure

The dataset (medical_dataset.csv) generated using gpt includes:

* disease_name
* symptoms
* causes
* remedies
* risk_factors
* prevention
* description
These fields are merged into a single text column during embedding.


      Ranking Engine (Already Handled by Pinecone)

* This project does not require a custom ranking engine implementation, because:
Pinecone automatically provides ranking
When you perform a query:
index.query(vector=embedding, top_k=5)
Pinecone:
Computes vector similarity (cosine distance)
Ranks all vectors in the index
Returns the top-K most relevant results
Includes a score that represents similarity strength
This means the search quality, ordering, and ranking are already handled internally by the vector database.


      Embedding + Upsert Pipeline

embed_upsert.py:
* Loads the medical CSV dataset
* Creates a combined text field
* Generates embeddings using SentenceTransformers
* Uploads embeddings to Pinecone via batch upsert

Run:
python src/embeddings/embed_upsert.py

      Semantic Search Engine

search_engine.py:
* Converts user query → embedding
* Searches Pinecone index
* Retrieves top-k ranked results with scores
* Returns metadata (the medical text)

      Query Enhancement Using Groq

query_intelligence.py:
* Rewrites user queries
* Adds medical context
* Greatly boosts search accuracy
* This step is automatic in the UI.

      Streamlit User Interface

Features:
* Query input
* Automatic LLM-based enhancement
* Ability to view rewritten query
* Semantic search results
* Ranking scores from Pinecone

To launch:
streamlit run src/ui/app.py


      Environment Variables

Create .env:
* GROQ_API_KEY=your_groq_key
* PINECONE_API_KEY=your_pinecone_key

      Running the Entire System

1. Embed dataset + upload to Pinecone
python src/embeddings/embed_upsert.py

2. Launch Streamlit UI
streamlit run src/ui/app.py



