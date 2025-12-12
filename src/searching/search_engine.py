import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.query_enhance.query_intelligence import enhance_query  

# Connect to Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Your index name
index = pc.Index("med-db")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Search function for retrieving relevant documents from Pinecone based on enhanced query
def search(query, top_k=5):
    
    # Enhance the query
    query = enhance_query(query)
    print ("Enhanced Query:", query)
    
    # Convert query to embedding
    embedding = model.encode([query])[0].tolist()
   
    # Search in Pinecone
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Prepare readable results
    outputs = []
    for match in result.matches:
        outputs.append({
            "score": match.score,
            "text": match.metadata.get("text", "")
        })

    return outputs
