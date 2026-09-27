from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric, AnswerRelevancyMetric)


from golden_dataset import golden_cases
from rag_generate import generate_answer

test_cases = []

for case in golden_cases:
  query = case["input"]
  expected_output = case["expected_output"]
  category=case["category"]
  
  test_id = case["id"]
  category = case["category"]
  expected_behavior = case["expected_behavior"]

  print("\n===============Test Case Details==========================")
  print(f"Test ID: {test_id}")
  print(f"Category: {category}")
  print(f"Expected behavior: {expected_behavior}")
  print("========================================")

  if category != "in_domain":
      print(
            "SKIPPED: This test requires a dedicated abstention evaluation."
        )
      continue;

  actual_output, results  = generate_answer(query=query, k=3)

  retrieval_context = [
    doc.page_content for doc in results
  ]

  test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    expected_output = expected_output,
    retrieval_context = retrieval_context
  )

  test_cases.append(test_case)

  print("\n============= GOLDEN CASEs In Execution =============")
  print("Input:")
  print(query)

  print("\nActual Output:")
  print(actual_output)

  print("\nRetrieved Context:")
  for i, doc in enumerate(results, 1):
      print(f"\n--- Context {i} ---")
      print(doc.page_content)

faithfulness_metric = FaithfulnessMetric(
  threshold=0.5,
  include_reason=True
)

answer_relevancy_metric = AnswerRelevancyMetric(
  threshold=0.5,
  include_reason=True
)

contextual_precision_metric = ContextualPrecisionMetric(
   threshold=0.5, include_reason=True
)

contextual_recall_metric = ContextualRecallMetric(
   threshold=0.5,
   include_reason=True
)

evaluate(test_cases=test_cases, metrics=[contextual_precision_metric, contextual_recall_metric, faithfulness_metric, answer_relevancy_metric])