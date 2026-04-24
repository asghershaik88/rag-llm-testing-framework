# Monitoring/drift_detector.py

def detect_drift(current_results, history_results):
    if not history_results:
        return "no_history"

    def avg_score(results):
        valid = [r for r in results if r["faithfulness"] is not None]
        if not valid:
            return 0
        return sum(r["faithfulness"] for r in valid) / len(valid)

    current_avg = avg_score(current_results)
    history_avg = avg_score(history_results)

    if current_avg < history_avg * 0.8:
        return "drift_detected"

    return "stable"