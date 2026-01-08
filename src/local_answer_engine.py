def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    """
    Deterministic, local answer generator using retrieved knowledge.
    No LLM required.
    """

    if not retrieved_docs:
        return (
            "I don’t have enough relevant information yet. "
            "Try asking about sleep, stress, diet, or activity patterns."
        )

    answer = "Here’s what may help based on known health patterns:\n\n"

    for i, doc in enumerate(retrieved_docs[:3], start=1):
        answer += f"{i}. {doc.strip()}\n\n"

    answer += (
        " These suggestions are informational and not a medical diagnosis. "
        "Small, consistent lifestyle changes often lead to meaningful improvements."
    )

    return answer
