from pathlib import Path
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


def main():

    print("=" * 60)
    print("TERRASPECTRA - FINAL TEST EVALUATION")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # LOAD TEST DATA
    # ------------------------------------------------------

    print("\nLoading test data...")

    X_test = np.load(TEST_DIR / "X_test.npy")
    y_test = np.load(TEST_DIR / "y_test.npy")

    print(f"X_test original shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Add channel dimension for Conv3D
    X_test = X_test.astype(np.float32)
    X_test = np.expand_dims(X_test, axis=-1)

    print(f"X_test prepared shape: {X_test.shape}")

    # ------------------------------------------------------
    # LOAD BEST MODEL
    # ------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found:\n{MODEL_PATH}"
        )

    print("\nLoading best Improved 3D-CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

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

    report_path = RESULTS_DIR / "classification_report_3dcnn.csv"

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

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_csv_path = RESULTS_DIR / "confusion_matrix_3dcnn.csv"
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
        cmap="Blues",
        values_format="d"
    )

    plt.title("TerraSpectra Improved 3D-CNN - Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    cm_plot_path = RESULTS_DIR / "confusion_matrix_3dcnn.png"

    plt.savefig(
        cm_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved confusion matrix plot: {cm_plot_path}")

    # ------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------

    summary_path = RESULTS_DIR / "final_test_results_3dcnn.txt"

    with open(summary_path, "w", encoding="utf-8") as f:

        f.write("TERRASPECTRA - FINAL TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")

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