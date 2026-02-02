def agent_decision(text, sentiment, confidence, rag):
    agent_log = []
    agent_log.append(f"Sentiment: {sentiment}")
    agent_log.append(f"Confidence: {confidence}")

    if confidence == "High":
        agent_log.append("Decision: High confidence → RAG not required")
        return {
            "explanation": (
                "All sentiment models agree on the prediction, "
                "resulting in high confidence."
            ),
            "rag_used": False,
            "agent_log": agent_log
        }

    agent_log.append("Decision: Medium/Low confidence → RAG activated")
    explanation = rag.explain(text)

    return {
        "explanation": explanation,
        "rag_used": True,
        "agent_log": agent_log
    }



