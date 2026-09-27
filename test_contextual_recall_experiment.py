from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

query="How dod I use Dark Mode?"

expected_answer = (
    "To use Dark Mode, open Settings and tap Display. "
    "Select Dark to apply the dark theme. "
    "Dark mode settings can be used to customize when and where "
    "Dark mode is applied, including Sunset to sunrise or "
    "a Custom schedule."
)

# Intentionally lets put incomplete retrieval context.
# It contains Settings -> Display and Dark,
# but does NOT contain scheduling information of dark mode.
retrieval_context = [
    """
    Dark mode
    Dark mode allows you to switch to a darker theme.
    From Settings, tap Display.
    Light: Apply a light color theme to your device.
    Dark: Apply a dark color theme to your device.
    """
]

test_case = LLMTestCase(
    input=query,
    expected_output=expected_answer,
    retrieval_context=retrieval_context
)


contextual_recall_metric = ContextualRecallMetric(
    threshold=0.5,
    include_reason=True
)


evaluate(
    test_cases=[test_case],
    metrics=[contextual_recall_metric]
)