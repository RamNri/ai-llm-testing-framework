import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="How do I use Dark Mode?"
)

embedding = response.data[0].embedding

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")