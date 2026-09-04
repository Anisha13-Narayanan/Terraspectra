from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# TERRASPECTRA - FINAL 3D-CNN TEST EVALUATION
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

TEST_DIR = PROJECT_ROOT / "data" / "patches" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODELS_DIR / "best_improved_3dcnn.keras"

CLASS_NAMES = [
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum",
    "Healthy"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a TerraSpectra 3D-CNN on a saved test split."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to a Keras model (default: best_improved_3dcnn.keras).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_DIR,
        help="Directory containing X_test.npy and y_test.npy.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for evaluation artifacts.",
    )
    return parser.parse_args()


def load_test_data(test_dir):
    X_path = test_dir / "X_test.npy"
    y_path = test_dir / "y_test.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Test files not found in {test_dir}. "
            "Expected X_test.npy and y_test.npy."
        )

    X_test = np.load(X_path)
    y_test = np.asarray(np.load(y_path)).reshape(-1)

    if X_test.ndim != 4 or X_test.shape[1:] != (32, 32, 30):
        raise ValueError(
            f"Expected test patches with shape (N, 32, 32, 30), "
            f"got {X_test.shape}."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"X/y sample mismatch: {len(X_test)} patches and "
            f"{len(y_test)} labels."
        )

    if not np.isfinite(X_test).all():
        raise ValueError("Test patches contain NaN or infinite values.")

    if not np.isin(y_test, np.arange(len(CLASS_NAMES))).all():
        raise ValueError(
            f"Labels must be integers from 0 to {len(CLASS_NAMES) - 1}; "
            f"got {np.unique(y_test).tolist()}."
        )

    return X_test.astype(np.float32), y_test.astype(np.int64)


def main():
    args = parse_args()

    print("=" * 60)
    print("TERRASPECTRA - FINAL TEST EVALUATION")
    print("=" * 60)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # LOAD TEST DATA
    # ------------------------------------------------------

    print("\nLoading test data...")

    X_test, y_test = load_test_data(args.test_dir)

    print(f"X_test original shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(
        f"X statistics: min={X_test.min():.4f}, "
        f"max={X_test.max():.4f}, mean={X_test.mean():.4f}, "
        f"std={X_test.std():.4f}"
    )
    print(
        "Class counts: "
        f"{np.bincount(y_test, minlength=len(CLASS_NAMES)).tolist()}"
    )

    # ------------------------------------------------------
    # LOAD BEST MODEL
    # ------------------------------------------------------

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found:\n{args.model}"
        )

    print(f"\nLoading model: {args.model.name}...")
    model = tf.keras.models.load_model(args.model)

    print("Model loaded successfully.")

    # Conv3D expects a channel dimension; the 2D ViT uses the PCA bands
    # directly as channels.
    expected_rank = len(model.input_shape)
    if expected_rank == X_test.ndim + 1:
        X_test = np.expand_dims(X_test, axis=-1)
    elif expected_rank != X_test.ndim:
        raise ValueError(
            f"Model expects input rank {expected_rank}, "
            f"but test data has rank {X_test.ndim}."
        )

    print(f"X_test prepared shape: {X_test.shape}")

    # ------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------

    print("\nEvaluating on untouched test data...")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # ------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)

    # Verify sklearn accuracy
    sklearn_accuracy = accuracy_score(y_test, y_pred)

    print(
        f"Verified Accuracy (sklearn): "
        f"{sklearn_accuracy:.4f}"
    )

    # ------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_path = args.results_dir / "classification_report_3dcnn.csv"

    report_df.to_csv(report_path)

    print("\nCLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print(f"Saved report: {report_path}")

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    print("\nGenerating confusion matrix...")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_csv_path = args.results_dir / "confusion_matrix_3dcnn.csv"
    cm_df.to_csv(cm_csv_path)

    print("\nCONFUSION MATRIX")
    print("=" * 60)
    print(cm)

    print(f"\nSaved confusion matrix CSV: {cm_csv_path}")

    # Plot
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    disp.plot(
        ax=ax,
        cmap="viridis",
        values_format="d"
    )

    plt.title("TerraSpectra Improved 3D-CNN - Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    cm_plot_path = args.results_dir / "confusion_matrix_3dcnn.png"

    plt.savefig(
        cm_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved confusion matrix plot: {cm_plot_path}")

    # ------------------------------------------------------
    # SAVE PER-SAMPLE PREDICTIONS
    # ------------------------------------------------------

    predictions_df = pd.DataFrame({
        "sample_index": np.arange(len(y_test)),
        "true_label": y_test,
        "true_class": [CLASS_NAMES[label] for label in y_test],
        "predicted_label": y_pred,
        "predicted_class": [CLASS_NAMES[label] for label in y_pred],
        "confidence": probabilities.max(axis=1),
    })

    for label, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"probability_{class_name}"] = probabilities[:, label]

    predictions_path = args.results_dir / "predictions_3dcnn.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions: {predictions_path}")

    confidence_rows = [{
        "group": "overall",
        "sample_count": len(predictions_df),
        "mean_confidence": predictions_df["confidence"].mean(),
        "mean_confidence_correct": predictions_df.loc[
            predictions_df["true_label"] == predictions_df["predicted_label"],
            "confidence",
        ].mean(),
        "accuracy": sklearn_accuracy,
    }]

    for label, class_name in enumerate(CLASS_NAMES):
        class_predictions = predictions_df[
            predictions_df["predicted_label"] == label
        ]
        confidence_rows.append({
            "group": class_name,
            "sample_count": len(class_predictions),
            "mean_confidence": class_predictions["confidence"].mean(),
            "mean_confidence_correct": class_predictions.loc[
                class_predictions["true_label"] == label,
                "confidence",
            ].mean(),
            "accuracy": (
                (class_predictions["true_label"] == label).mean()
                if len(class_predictions) > 0
                else np.nan
            ),
        })

    confidence_path = args.results_dir / "confidence_summary.csv"
    pd.DataFrame(confidence_rows).to_csv(confidence_path, index=False)
    print(f"Saved confidence summary: {confidence_path}")

    # ------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------

    summary_path = args.results_dir / "final_test_results_3dcnn.txt"

    with open(summary_path, "w", encoding="utf-8") as f:

        f.write("TERRASPECTRA - FINAL TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Model: {args.model}\n")
        f.write(f"Test data: {args.test_dir}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Accuracy Percent: {test_accuracy * 100:.2f}%\n")

        f.write("\nCLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")

        f.write(
            classification_report(
                y_test,
                y_pred,
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )

    print(f"\nSaved final summary: {summary_path}")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()