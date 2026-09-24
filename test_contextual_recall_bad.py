from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

query="How do I use Dark Mode?"
expected_answer= (
    "To use Dark Mode, open Settings, go to Display, "
    "and select Dark mode."
)

test_case = LLMTestCase(
  input=query,
  actual_output=expected_answer,
  expected_output=expected_answer,
  retrieval_context=[
    "Open the Camera application.",
    "Configure shooting modes and camera settings.",
    "You can record videos and edit pictures in Gallery."
  ]
)

metric= ContextualRecallMetric(
  threshold=0.5,
)


evaluate(test_cases=[test_case], metrics=[metric])