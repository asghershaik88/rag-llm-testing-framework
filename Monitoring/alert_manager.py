# Monitoring/alert_manager.py

def generate_alert(failure_type, drift_status):
    if failure_type == "hallucination":
        return "🚨 HIGH ALERT: Hallucination detected"

    if drift_status == "drift_detected":
        return "⚠️ MEDIUM ALERT: Performance drift detected"

    if failure_type == "retrieval_issue":
        return "⚠️ LOW ALERT: Retrieval issue"

    return "✅ OK"