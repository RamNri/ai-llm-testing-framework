from dotenv import load_dotenv
from openai import OpenAI
from simple_rag import load_vector_store, retrieve_relevant_docs

load_dotenv(override=True)

client = OpenAI()

def generate_answer(query, k=3):


  vector_store = load_vector_store()
  results = retrieve_relevant_docs(vector_store = vector_store, query=query, k=k)
  context = "\n\n".join(doc.page_content for doc in results)

  prompt = f"""
  Answer the question using only the information provided in the context.

  Context:
  {context}
  
  Question:
  {query}

"""
  response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
  )

  return response.output_text, results

if __name__ == "__main__":
  query = "How do I use Dark Mode?"

  answer, results = generate_answer(query=query, k=3)

  print("\n========Retrieved Content===========")

  for i, doc in enumerate(results, 1):
    print(f"\n=====Retrieved Context {i} ----")
    print(doc.page_content)

  print("\n=======Generated Answer==========")
  print(answer)  