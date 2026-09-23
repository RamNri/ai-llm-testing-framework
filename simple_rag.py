import os #File folder handling
import pymupdf  #PDF extraction
from dotenv import load_dotenv
 
from langchain_google_genai import GoogleGenerativeAIEmbeddings #text -> vector
from langchain_community.vectorstores import FAISS #vector similarity search
from langchain_text_splitters import RecursiveCharacterTextSplitter #break text into chunks
from langchain_core.documents import Document  #Document and meta data

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
  raise ValueError("Google api key was not found in the environment")

def extract_text_from_pdf(pdf_path):
  text = ""

  with pymupdf.open(pdf_path) as doc:
    for page in doc:
      text += page.get_text("text") + "\n"

  return text

def create_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  chunks = text_splitter.split_text(text) 
  return chunks

def create_embeddings(chunks):
  embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
  )
  vectors = embeddings.embed_documents(chunks)
  return vectors

def create_vector_store(chunks):
  documents = [
    Document(page_content=chunk)
    for chunk in chunks
  ]

  embeddings = GoogleGenerativeAIEmbeddings(
      model="models/gemini-embedding-001",
      google_api_key=api_key
    )

  vector_store = FAISS.from_documents(documents, embeddings)
  vector_store.save_local("faiss_index")
  return vector_store

def retrieve_relevant_docs(vector_store, query, k=3):
  results = vector_store.similarity_search(query, k=k)
  return results

if __name__ == "__main__":
  pdf_path = "data/documents/s22_manual.pdf"

  text = extract_text_from_pdf(pdf_path=pdf_path)
  chunks = create_chunks(text)
  print(f"Total chunks created: {len(chunks)}")

  vector_store = create_vector_store(chunks=chunks[:20])

  query = "How do I use Dark Mode?"
  results = retrieve_relevant_docs(vector_store=vector_store, query=query)
  for i, doc in enumerate(results, 1):
    print(f"\n---Result {i}---")
    print(doc.page_content)