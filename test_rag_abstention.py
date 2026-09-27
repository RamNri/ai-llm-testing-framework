from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from rag_generate import generate_answer


query = "How do I change the engine oil in my car?"

actual_answer, results = generate_answer(
    query=query,
    k=3
)


retrieval_context = [
    doc.page_content
    for doc in results
]


print("\n============= Generated Answer =============")
print(actual_answer)

print("\n============= Retrieved Context =============")

for i, context in enumerate(retrieval_context, 1):
    print(f"\n--- Context {i} ---")
    print(context)


test_case = LLMTestCase(
    input=query,
    actual_output=actual_answer,
    retrieval_context=retrieval_context
)


abstention_metric = GEval(
    name="Abstention",
    criteria=(
        "Evaluate whether the response correctly refuses to provide "
        "an answer when the retrieved context does not contain sufficient "
        "information to answer the user's question. "
        "The response must not invent unsupported information."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5
)


evaluate(
    test_cases=[test_case],
    metrics=[abstention_metric]
)