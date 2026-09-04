from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from train_shared_pca_3dcnn import build_model
except ModuleNotFoundError:
    from src.train_shared_pca_3dcnn import build_model


PROJECT_ROOT = Path(r"E:\Terraspectra")
DATA_DIR = PROJECT_ROOT / "data" / "patches_shared_pca"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.0005
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    X_train = np.load(DATA_DIR / "train" / "X_train.npy").astype(np.float32)
    y_train = np.load(DATA_DIR / "train" / "y_train.npy").astype(np.int32)
    X_val = np.load(DATA_DIR / "val" / "X_val.npy").astype(np.float32)
    y_val = np.load(DATA_DIR / "val" / "y_val.npy").astype(np.int32)
    return X_train[..., None], y_train, X_val[..., None], y_val


def augment_training_data(X_train, y_train):
    variants = [
        X_train,
        np.flip(X_train, axis=1),
        np.flip(X_train, axis=2),
        np.flip(X_train, axis=(1, 2)),
    ]
    return np.concatenate(variants, axis=0), np.tile(y_train, 4)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_data()
    X_train, y_train = augment_training_data(X_train, y_train)
    print(f"Augmented train X: {X_train.shape}")
    print(f"Augmented train y: {y_train.shape}")

    model = build_model(X_train.shape[1:])
    model.optimizer.learning_rate.assign(LEARNING_RATE)

    best_model_path = MODELS_DIR / "best_shared_pca_augmented_3dcnn.keras"
    history_path = RESULTS_DIR / "training_history_shared_pca_augmented_3dcnn.csv"

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        shuffle=True,
        verbose=1,
    )

    model.save(MODELS_DIR / "final_shared_pca_augmented_3dcnn.keras")
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best model: {best_model_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
