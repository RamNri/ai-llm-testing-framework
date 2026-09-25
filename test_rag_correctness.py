from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from rag_generate import generate_answer

query = "How do I use Dark Mode?"

expected_answer = (
    "To use Dark Mode, open Settings and tap Display. "
    "Select Dark to apply the dark theme. "
    "Dark mode settings can be used to customize when and where "
    "Dark mode is applied, including Sunset to sunrise or a Custom schedule."
)

actual_answer, results = generate_answer(
    query=query,
    k=3
)

print("\n============= Generated Answer =============")
print(actual_answer)

retrieval_context = [
    doc.page_content
    for doc in results
]

test_case = LLMTestCase(
    input=query,
    actual_output=actual_answer,
    expected_output=expected_answer,
    retrieval_context=retrieval_context
)

correctness_metric = GEval(
    name="Answer Correctness",
    criteria=(
        "Evaluate whether the actual answer correctly answers the input question "
        "and is consistent with the expected answer. "
        "Consider semantic correctness rather than exact wording. "
        "Do not penalize the answer merely because it contains additional "
        "relevant information."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5
)

evaluate(
    test_cases=[test_case],
    metrics=[correctness_metric]
)