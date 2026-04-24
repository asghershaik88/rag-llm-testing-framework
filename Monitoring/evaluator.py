import asyncio
import sys
from pathlib import Path
import json

from deepeval.benchmarks import results

from Ragas_evaluations.context_recall import context_recall_score
from Ragas_evaluations.faithfulness import faithfulness_score
from Monitoring.failureclasiifier import classify_failure
from Monitoring.drift_detector import detect_drift
from Monitoring.alert_manager import generate_alert


def load_json_logs():
    logs = []
    path = Path(__file__).parent / "logs.jsonl"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            logs.append(json.loads(line))

    return logs


def evaluate(query, context, response):
    try:
        faithfulness = faithfulness_score(query, response, context)
        context_recall = context_recall_score(query, response, context)

        print("Faithfulness:", faithfulness)
        print("Context Recall:", context_recall)

        return {
            "faithfulness": faithfulness,
            "context_recall": context_recall
        }

    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "faithfulness": None,
            "context_recall": None
        }


if __name__ == "__main__":
    logs = load_json_logs()

    print("TOTAL LOGS:", len(logs))

    history_results = []  # simulate past runs

    for i, log in enumerate(logs):
        print(f"\nEvaluating log {i}")

        scores = evaluate(
            log["query"],
            log["context"],
            log["response"]
        )

        # Step 1: classify failure
        failure_type = classify_failure(log, scores)

        # Step 2: drift detection (compare with history)
        drift_status = detect_drift([scores], history_results)

        # Step 3: alert generation
        alert = generate_alert(failure_type, drift_status)

        print("Failure Type:", failure_type)
        print("Drift Status:", drift_status)
        print("Alert:", alert)

        results.append({
            "scores": scores,
            "failure_type": failure_type,
            "drift_status": drift_status,
            "alert": alert
        })

        # update history
        history_results.append(scores)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())