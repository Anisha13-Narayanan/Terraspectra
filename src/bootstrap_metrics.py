from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(r"E:\Terraspectra")
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "shared_pca_3dcnn" / "predictions_3dcnn.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "shared_pca_3dcnn" / "bootstrap_metrics.json"
NUM_CLASSES = 5
RESAMPLES = 10000
SEED = 42


def main():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    true_labels = predictions["true_label"].to_numpy(dtype=np.int64)
    predicted_labels = predictions["predicted_label"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(SEED)

    accuracy_samples = np.empty(RESAMPLES, dtype=np.float64)
    f1_samples = np.empty(RESAMPLES, dtype=np.float64)
    sample_count = len(true_labels)

    for index in range(RESAMPLES):
        selected = rng.integers(0, sample_count, size=sample_count)
        sampled_true = true_labels[selected]
        sampled_predicted = predicted_labels[selected]
        accuracy_samples[index] = (sampled_true == sampled_predicted).mean()
        f1_samples[index] = f1_score(
            sampled_true,
            sampled_predicted,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )

    summary = {
        "prediction_file": str(PREDICTIONS_PATH),
        "samples": sample_count,
        "resamples": RESAMPLES,
        "seed": SEED,
        "accuracy": float((true_labels == predicted_labels).mean()),
        "accuracy_ci_95": [
            float(np.percentile(accuracy_samples, 2.5)),
            float(np.percentile(accuracy_samples, 97.5)),
        ],
        "macro_f1": float(f1_score(
            true_labels,
            predicted_labels,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )),
        "macro_f1_ci_95": [
            float(np.percentile(f1_samples, 2.5)),
            float(np.percentile(f1_samples, 97.5)),
        ],
        "note": "Patch-level bootstrap; source-file-level validation is still recommended.",
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved bootstrap metrics: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
