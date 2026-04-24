import json
from pathlib import Path

from Ragas_evaluations.faithfulness import faithfulness_score


def load_log_file():
    logs=[]
    path=Path(__file__).parent/"logs.jsonl"
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            logs.append(json.loads(line))
    return logs

def evaluate(query,response,context):
    Faithfulness_score = faithfulness_score(query,response,context)
    metrics={
        "Faithfulness_score":Faithfulness_score,
            }
    return metrics

if __name__ == "__main__":
    logs = load_log_file()
    results=[]
    for log in logs:
        result = evaluate(log["query"],log["response"],log["context"])
        results.append(result)
        print(result)










