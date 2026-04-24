

def classify_failure(log, scores):
    if scores["faithfulness"] is None:
        return "evaluation_error"

    if scores["faithfulness"] < 0.5:
        return "hallucination"

    if scores["context_recall"] < 0.5:
        return "retrieval_issue"

    return "pass"