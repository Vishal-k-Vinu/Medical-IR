import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

# Function to enhance medical search queries
def enhance_query(query: str) -> str:
    prompt = f'''Enhance the following medical search query to improve retrieval accuracy. Original Query: {query}
    example:
    Original Query: "lung infection cause"
    Enhanced Query: "What are the causes of lung infections such as pneumonia?"
     it should replace the query with enhanced query.
    '''
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that enhances medical search queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip()