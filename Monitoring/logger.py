import datetime
import json
from pathlib import Path



LOG_FILE = "Monitoring/logs.jsonl"

path = Path(__file__).parent.parent /"Monitoring/logs.jsonl"
def ensure_log_file():
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def log_interaction(query, context, response):
    ensure_log_file()

    clean_response = response.content if hasattr(response, 'content') else str(response)

    log_entry = {
        "query": query,
        "context": context,
        "response": clean_response,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


    with open(path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")