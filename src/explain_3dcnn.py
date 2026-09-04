from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom


PROJECT_ROOT = Path(r"E:\Terraspectra")
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_shared_pca_3dcnn.keras"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "patches_shared_pca" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "explainability"

CLASS_NAMES = [
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum",
    "Healthy",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spatial and spectral Grad-CAM explanations."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_patch(test_dir, index):
    X = np.load(test_dir / "X_test.npy").astype(np.float32)
    y = np.asarray(np.load(test_dir / "y_test.npy")).reshape(-1)

    if X.ndim != 4 or X.shape[1:] != (32, 32, 30):
        raise ValueError(f"Expected (N, 32, 32, 30), got {X.shape}")
    if len(X) != len(y):
        raise ValueError("Test feature and label counts do not match")
    if not 0 <= index < len(X):
        raise IndexError(f"Index must be between 0 and {len(X) - 1}")

    return X[index], int(y[index])


def make_attribution(model, patch, class_index):
    input_tensor = tf.Variable(patch[None, ..., None])
    with tf.GradientTape() as tape:
        predictions = model(input_tensor, training=False)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, input_tensor)
    cam = tf.abs(gradients[0, ..., 0]).numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam, predictions[0].numpy()


def save_explanation(output_dir, patch, true_label, predicted_label, confidence, cam):
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_cam = cam.mean(axis=2)
    spatial_cam = zoom(spatial_cam, (32 / spatial_cam.shape[0], 32 / spatial_cam.shape[1]))
    spatial_cam = spatial_cam[:32, :32]

    spectral_importance = cam.mean(axis=(0, 1))
    spectral_importance = zoom(
        spectral_importance,
        30 / spectral_importance.shape[0],
    )[:30]
    if spectral_importance.max() > spectral_importance.min():
        spectral_importance = (
            (spectral_importance - spectral_importance.min())
            / (spectral_importance.max() - spectral_importance.min())
        )

    np.save(output_dir / "cam_3d.npy", cam)
    np.save(output_dir / "spatial_heatmap.npy", spatial_cam)
    np.save(output_dir / "spectral_importance.npy", spectral_importance)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(spatial_cam, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("Spatial importance")
    axes[0].set_xlabel("Patch column")
    axes[0].set_ylabel("Patch row")
    axes[1].plot(np.arange(1, 31), spectral_importance, color="teal")
    axes[1].set_title("PCA component importance")
    axes[1].set_xlabel("PCA component")
    axes[1].set_ylabel("Relative importance")
    fig.suptitle(
        f"Predicted: {CLASS_NAMES[predicted_label]} "
        f"({confidence * 100:.1f}%) | True: {CLASS_NAMES[true_label]}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "explanation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "true_label": true_label,
        "true_class": CLASS_NAMES[true_label],
        "predicted_label": predicted_label,
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": float(confidence),
        "cam_shape": list(cam.shape),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    patch, true_label = load_patch(args.test_dir, args.index)
    model = tf.keras.models.load_model(args.model)
    cam, probabilities = make_attribution(
        model,
        patch,
        int(np.argmax(model(patch[None, ..., None], training=False).numpy()[0])),
    )

    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])
    save_explanation(
        args.output_dir,
        patch,
        true_label,
        predicted_label,
        confidence,
        cam,
    )
    print(f"True class: {CLASS_NAMES[true_label]}")
    print(f"Predicted class: {CLASS_NAMES[predicted_label]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Saved explanation: {args.output_dir}")


if __name__ == "__main__":
    main()