import json

import numpy as np
from scipy.stats import ttest_ind


with open("hallucination_baseline.json") as f:
    baseline_scores=json.load(f)

with open("hallucination_newscores.json") as f:
    new_scores=json.load(f)


def run_ttest(baseline_scores,new_scores):
    t_stat,p_value=ttest_ind(baseline_scores,new_scores)

    baseline_mean = np.mean(baseline_scores)
    new_mean = np.mean(new_scores)

    print("Baseline Mean:", baseline_mean)
    print("New Mean:", new_mean)
    print("P-value:", p_value)


    if p_value < 0.05:
        if new_mean < baseline_mean:
            print("✅ Significant improvement")
        else:
            print("❌ Significant regression")
    else:
        print("⚠️ No significant difference")

    return t_stat, p_value



run_ttest(baseline_scores, new_scores)