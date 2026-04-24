import json
from pathlib import Path



def load_data():
    path = Path(__file__).parent.parent/"data/golden_data.json"
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    return data







