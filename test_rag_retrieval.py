from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from simple_rag import retrieve_relevant_docs, load_vector_store


vector_store = load_vector_store()
query = "How do I change the engine oil in my car?"

#How do I use Dark Mode?

results = retrieve_relevant_docs(vector_store=vector_store, query=query, k=3)

print("\n=============Retrieved Context ================")
for i, doc in enumerate(results, 1):
  print(f"\n--- Retrieved Context {i} ---")
  print(doc.page_content)

retrieval_context = [doc.page_content for doc in results]

test_case = LLMTestCase(
  input=query,
  actual_output="Dark Mode can be enabled from the display settings.",
  expected_output="Dark Mode can be enabled from the display settings.",
  retrieval_context=retrieval_context
)

metric = ContextualPrecisionMetric(
  threshold=0.5,
  include_reason=True
)

evaluate(test_cases=[test_case], metrics=[metric])