from deepeval import evaluate
from deepeval.metrics import (ContextualPrecisionMetric, ContextualRecallMetric)
from deepeval.test_case import LLMTestCase


query = "How do I use Dark Mode?"

expected_answer = (
    "To use Dark Mode, open Settings and tap Display. "
    "Select Dark to apply the dark theme. "
    "Dark mode settings can be used to customize when and where "
    "Dark mode is applied, including Sunset to sunrise or "
    "a Custom schedule."
)


# Deliberately noisy retrieval context.
# The relevant Dark Mode information is present,
# but unrelated information is also included.
retrieval_context = [

    # Relevant
    """
    Dark mode
    Dark mode allows you to switch to a darker theme.
    From Settings, tap Display.
    Light: Apply a light color theme to your device.
    Dark: Apply a dark color theme to your device.
    """,

    # Irrelevant
    """
    Camera
    From the Camera app, tap Settings to customize camera options,
    shooting modes, and picture quality.
    """,

    # Irrelevant
    """
    Battery
    Battery settings allow you to configure power saving options
    and monitor battery usage.
    """,

    # Relevant
    """
    Dark mode settings: Customize when and where Dark mode is applied.
    Turn on as scheduled: Configure Dark mode for either
    Sunset to sunrise or Custom schedule.
    """
]


test_case = LLMTestCase(
    input=query,
    expected_output=expected_answer,
    retrieval_context=retrieval_context
)


contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.5,
    include_reason=True
)

contextual_recall_metric = ContextualRecallMetric(
    threshold=0.5,
    include_reason=True
)


evaluate(
    test_cases=[test_case],
    metrics=[
        contextual_precision_metric,
        contextual_recall_metric
    ]
)