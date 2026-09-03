from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# TERRASPECTRA - TRAINING HISTORY VISUALIZATION
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

RESULTS_DIR = PROJECT_ROOT / "results"

HISTORY_FILE = RESULTS_DIR / "training_history_improved_3dcnn.csv"


def main():

    print("=" * 60)
    print("TERRASPECTRA - TRAINING HISTORY VISUALIZATION")
    print("=" * 60)

    # ------------------------------------------------------
    # Load training history
    # ------------------------------------------------------

    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_FILE}"
        )

    history = pd.read_csv(HISTORY_FILE)

    print("\nTraining history loaded successfully.")
    print(f"Total epochs recorded: {len(history)}")

    print("\nColumns found:")
    print(list(history.columns))

    epochs = range(1, len(history) + 1)

    # ------------------------------------------------------
    # Accuracy Graph
    # ------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("TerraSpectra Improved 3D-CNN Accuracy")
    plt.legend()
    plt.grid(True)

    accuracy_path = (
        RESULTS_DIR /
        "improved_3dcnn_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nAccuracy graph saved: {accuracy_path}")

    # ------------------------------------------------------
    # Loss Graph
    # ------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TerraSpectra Improved 3D-CNN Loss")
    plt.legend()
    plt.grid(True)

    loss_path = (
        RESULTS_DIR /
        "improved_3dcnn_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Loss graph saved: {loss_path}")

    # ------------------------------------------------------
    # Best Epoch Information
    # ------------------------------------------------------

    best_val_epoch = history["val_accuracy"].idxmax() + 1
    best_val_accuracy = history["val_accuracy"].max()

    print("\n" + "=" * 60)
    print("BEST VALIDATION RESULT")
    print("=" * 60)
    print(f"Best validation epoch: {best_val_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    print("\nGRAPHS GENERATED SUCCESSFULLY")


if __name__ == "__main__":
    main()