from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Terraspectra")
RESULTS_DIR = PROJECT_ROOT / "results"

EVALUATIONS = {
    "Shared-PCA 3D-CNN": RESULTS_DIR / "shared_pca_3dcnn",
    "Shared-PCA ViT": RESULTS_DIR / "vit",
    "Shared-PCA Hybrid": RESULTS_DIR / "shared_pca_hybrid",
    "Shared-PCA Augmented 3D-CNN": RESULTS_DIR / "shared_pca_augmented",
}


def read_metrics(model_name, results_dir):
    report_path = results_dir / "classification_report_3dcnn.csv"
    confidence_path = results_dir / "confidence_summary.csv"

    if not report_path.exists():
        raise FileNotFoundError(f"Missing classification report: {report_path}")

    report = pd.read_csv(report_path, index_col=0)
    accuracy = float(report.loc["accuracy", "f1-score"])
    macro = report.loc["macro avg"]

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "precision_macro": float(macro["precision"]),
        "recall_macro": float(macro["recall"]),
        "f1_macro": float(macro["f1-score"]),
    }

    if confidence_path.exists():
        confidence = pd.read_csv(confidence_path)
        metrics["mean_confidence"] = float(
            confidence.loc[confidence["group"] == "overall", "mean_confidence"].iloc[0]
        )
    else:
        metrics["mean_confidence"] = float("nan")

    return metrics


def main():
    comparison = pd.DataFrame(
        [read_metrics(name, path) for name, path in EVALUATIONS.items()]
    ).sort_values("accuracy", ascending=False)

    output_path = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved comparison: {output_path}")


if __name__ == "__main__":
    main()