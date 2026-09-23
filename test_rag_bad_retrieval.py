from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase


test_case = LLMTestCase(
    input="How do I use Dark Mode?",

    actual_output="Dark Mode can be enabled from the display settings.",

    expected_output="Dark Mode can be enabled from the display settings.",

    retrieval_context=[
        "The camera supports various shooting modes and AR Zone features.",
        "You can play videos and share pictures and videos from Gallery.",
        "The device supports wireless power sharing."
    ]
)


metric = ContextualPrecisionMetric(
    threshold=0.5,
    include_reason=True
)


evaluate(
    test_cases=[test_case],
    metrics=[metric]
)