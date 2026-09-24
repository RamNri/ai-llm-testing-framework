from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

query="How do I use Dark Mode?"
expected_answer=(
  "To use Dark Mode, open Settings, go to Display, "
    "select Dark mode, and choose when you want it to activate."
    )

test_case = LLMTestCase(
  input=query,
  actual_output=expected_answer,
  expected_output=expected_answer,
  retrieval_context=[
    "Open Settings and tap Display.",
    "Under Display settings, tap Dark mode.",
    "You can select Dark mode to change the device appearance."
  ]
)

metric = ContextualRecallMetric(
  threshold=0.5
)

evaluate(test_cases=[test_case], metrics=[metric])