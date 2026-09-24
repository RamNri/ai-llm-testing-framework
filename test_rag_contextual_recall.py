from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from simple_rag import retrieve_relevant_docs, load_vector_store

vector_store = load_vector_store()
query = "How do I use Dark Mode?"
results = retrieve_relevant_docs(vector_store=vector_store, query=query, k=3)

print("\n============= Retrieved Context =============")

for i, doc in enumerate(results, 1):
  print(f"\n---Retrieved Context {i} ---")
  print(doc.page_content)

retrieval_context = [doc.page_content for doc in results]

expected_answer = (
  "To use Dark Mode, open Settings and tap Display. "
    "Select Dark to apply the dark theme. "
    "Dark mode settings can be used to customize when and where "
    "Dark mode is applied, including Sunset to sunrise or a Custom schedule."
)

test_case = LLMTestCase(
  input=query,
  actual_output=expected_answer,
  expected_output=expected_answer,
  retrieval_context=retrieval_context
)

metric = ContextualRecallMetric(
  threshold=0.5,
  include_reason=True
)

evaluate(test_cases=[test_case], metrics=[metric])