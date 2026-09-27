golden_cases = [
    {
        "id": "RAG-DARK-MODE-001",
        "category": "in_domain",
        "expected_behavior": "answer",
        "input": "How do I use Dark Mode?",
        "expected_output": (
            "To use Dark Mode, open Settings and tap Display. "
            "Select Dark to apply the dark theme. "
            "Dark mode settings can be used to customize when and where "
            "Dark mode is applied, including Sunset to sunrise or "
            "a Custom schedule."
        )
    },
    {
        "id": "RAG-BRIGHTNESS-001",
        "category": "in_domain",
        "expected_behavior": "answer",
        "input": "How do I adjust the screen brightness?",
        "expected_output": (
            "To adjust the screen brightness, open Settings and tap Display. "
            "Under Brightness, drag the Brightness slider to set the "
            "desired brightness level."
        )
    },
    {
        "id": "RAG-OOD-001",
        "category": "out_of_domain",
        "expected_behavior": "abstain",
        "input": "How do I change the engine oil in my car?",
        "expected_output": (
            "The provided context does not contain sufficient information "
            "to explain how to change car engine oil."
        )
    }
]