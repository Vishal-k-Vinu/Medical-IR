import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os, sys

# Load Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "med-db"

# load dataset
df = pd.read_csv("data/medical_dataset.csv") 

# combining relevant columns into a single text field
def combine_columns(row):
    return (
        f"Disease: {row['disease_name']}. "
        f"Symptoms: {row['symptoms']}. "
        f"Causes: {row['causes']}. "
        f"Remedies: {row['remedies']}. "
        f"Risk Factors: {row['risk_factors']}. "
        f"Prevention: {row['prevention']}. "
        f"Description: {row['description']}."
    )

# Create 'text' and 'id' columns for embedding
df["text"] = df.apply(combine_columns, axis=1)
df["id"] = df.index.astype(str)  

# Prepare data for embedding
texts = df["text"].tolist()
ids = df["id"].tolist()

# connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Starting embedding + upsert...")

# Process and upsert in batches
batch_size = 256
for start in tqdm(range(0, len(texts), batch_size)):
    end = start + batch_size

    batch_texts = texts[start:end]
    batch_ids = ids[start:end]

    # Generate embeddings
    emb = model.encode(batch_texts, batch_size=32)

    # Prepare vectors for Pinecone
    vectors = [
        (batch_ids[i], emb[i].tolist(), {"text": batch_texts[i]})
        for i in range(len(batch_texts))
    ]

    # Upsert into Pinecone
    index.upsert(vectors=vectors)

print("All embeddings uploaded to Pinecone.")
