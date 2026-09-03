from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, layers, models, regularizers


# ==========================================================
# TERRASPECTRA - IMPROVED 3D-CNN TRAINING
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

PATCHES_DIR = PROJECT_ROOT / "data" / "patches"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0005
SEED = 42


# ==========================================================
# 1. REPRODUCIBILITY
# ==========================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

def load_dataset(split_name):
    """Load features and labels for a dataset split."""

    split_dir = PATCHES_DIR / split_name

    X_path = split_dir / f"X_{split_name}.npy"
    y_path = split_dir / f"y_{split_name}.npy"

    X = np.load(X_path)
    y = np.load(y_path)

    print("\n" + "=" * 60)
    print(f"LOADED {split_name.upper()} DATA")
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")

    return X, y


# ==========================================================
# 3. PREPARE DATA FOR 3D-CNN
# ==========================================================

def prepare_data(X):
    """
    Original shape:
    (samples, height, width, spectral_bands)

    3D-CNN shape:
    (samples, height, width, spectral_depth, channels)
    """

    X = X.astype(np.float32)

    # Add channel dimension
    X = np.expand_dims(X, axis=-1)

    return X


# ==========================================================
# 4. BUILD IMPROVED 3D-CNN
# ==========================================================

def conv_block(x, filters, dropout_rate=0.0):
    """3D convolution block with Batch Normalization."""

    x = layers.Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def build_3dcnn(input_shape, num_classes):

    inputs = layers.Input(shape=input_shape)

    # ------------------------------------------------------
    # Block 1
    # ------------------------------------------------------
    x = conv_block(
        inputs,
        filters=16,
        dropout_rate=0.10
    )

    x = layers.MaxPool3D(
        pool_size=(2, 2, 2)
    )(x)

    # ------------------------------------------------------
    # Block 2
    # ------------------------------------------------------
    x = conv_block(
        x,
        filters=32,
        dropout_rate=0.15
    )

    x = layers.MaxPool3D(
        pool_size=(2, 2, 2)
    )(x)

    # ------------------------------------------------------
    # Block 3
    # ------------------------------------------------------
    x = conv_block(
        x,
        filters=64,
        dropout_rate=0.20
    )

    # ------------------------------------------------------
    # Global feature extraction
    # ------------------------------------------------------
    x = layers.GlobalAveragePooling3D()(x)

    # ------------------------------------------------------
    # Classification head
    # ------------------------------------------------------
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.40)(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="TerraSpectra_Improved_3D_CNN"
    )

    return model


# ==========================================================
# 5. MAIN TRAINING
# ==========================================================

def main():

    print("=" * 60)
    print("TERRASPECTRA - IMPROVED 3D-CNN TRAINING")
    print("=" * 60)

    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------
    X_train, y_train = load_dataset("train")
    X_val, y_val = load_dataset("val")

    # ------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------
    X_train = prepare_data(X_train)
    X_val = prepare_data(X_val)

    print("\n" + "=" * 60)
    print("DATA AFTER 3D-CNN PREPARATION")
    print("=" * 60)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")

    # ------------------------------------------------------
    # Build model
    # ------------------------------------------------------
    input_shape = X_train.shape[1:]

    model = build_3dcnn(
        input_shape=input_shape,
        num_classes=NUM_CLASSES
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    model.summary()

    # ------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------

    best_model_path = MODELS_DIR / "best_improved_3dcnn.keras"

    training_callbacks = [

        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING IMPROVED 3D-CNN TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=training_callbacks,
        verbose=1
    )

    # ------------------------------------------------------
    # Save final model
    # ------------------------------------------------------

    final_model_path = MODELS_DIR / "final_improved_3dcnn.keras"

    model.save(final_model_path)

    # ------------------------------------------------------
    # Save training history
    # ------------------------------------------------------

    history_df = pd.DataFrame(history.history)

    history_path = (
        RESULTS_DIR /
        "training_history_improved_3dcnn.csv"
    )

    history_df.to_csv(history_path, index=False)

    # ------------------------------------------------------
    # Calculate best results
    # ------------------------------------------------------

    best_train_accuracy = max(history.history["accuracy"])
    best_val_accuracy = max(history.history["val_accuracy"])

    best_train_epoch = (
        np.argmax(history.history["accuracy"]) + 1
    )

    best_val_epoch = (
        np.argmax(history.history["val_accuracy"]) + 1
    )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("IMPROVED 3D-CNN TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best model saved: {best_model_path}")
    print(f"Final model saved: {final_model_path}")
    print(f"Training history: {history_path}")

    print("\nFINAL EPOCH RESULTS")
    print(
        f"Training accuracy: "
        f"{history.history['accuracy'][-1]:.4f}"
    )
    print(
        f"Validation accuracy: "
        f"{history.history['val_accuracy'][-1]:.4f}"
    )

    print("\nBEST RESULTS")
    print(
        f"Best training accuracy: "
        f"{best_train_accuracy:.4f} "
        f"(Epoch {best_train_epoch})"
    )
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f} "
        f"(Epoch {best_val_epoch})"
    )

    print("\nBaseline best validation accuracy: 0.4188")

    if best_val_accuracy > 0.4188:
        print("✓ Improved model performed better than baseline")
    else:
        print("⚠ Improved model did not beat the baseline yet")

    print("=" * 60)


if __name__ == "__main__":
    main()