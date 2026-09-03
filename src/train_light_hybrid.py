from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ==========================================================
# TERRASPECTRA - LIGHTWEIGHT 3D-CNN + TRANSFORMER HYBRID
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

PATCHES_DIR = PROJECT_ROOT / "data" / "patches"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0003
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_dataset(split_name):
    split_dir = PATCHES_DIR / split_name

    X = np.load(split_dir / f"X_{split_name}.npy")
    y = np.load(split_dir / f"y_{split_name}.npy")

    print(f"\nLoaded {split_name.upper()}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y


# ==========================================================
# PREPARE DATA
# ==========================================================

def prepare_data(X):
    X = X.astype(np.float32)
    return np.expand_dims(X, axis=-1)


# ==========================================================
# CNN BLOCK
# ==========================================================

def conv_block(x, filters, dropout_rate):

    x = layers.Conv3D(
        filters,
        kernel_size=(3, 3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3D(
        filters,
        kernel_size=(3, 3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dropout(dropout_rate)(x)

    return x


# ==========================================================
# LIGHTWEIGHT TRANSFORMER
# ==========================================================

def lightweight_transformer(inputs):

    projection_dim = 32

    # Attention block
    x1 = layers.LayerNormalization(epsilon=1e-6)(inputs)

    attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=16,
        dropout=0.25
    )(x1, x1)

    x2 = layers.Add()([inputs, attention])

    # Feed-forward block
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

    x3 = layers.Dense(
        64,
        activation="gelu"
    )(x3)

    x3 = layers.Dropout(0.25)(x3)

    x3 = layers.Dense(projection_dim)(x3)

    return layers.Add()([x2, x3])


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_light_hybrid(input_shape, num_classes):

    inputs = layers.Input(shape=input_shape)

    # 3D-CNN feature extraction
    x = conv_block(inputs, 16, 0.10)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    x = conv_block(x, 32, 0.15)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    # Final CNN features
    x = conv_block(x, 32, 0.20)

    # ------------------------------------------------------
    # Convert feature volume to Transformer tokens
    # ------------------------------------------------------
    # Current feature shape approximately:
    # (8, 8, 7, 32)

    x = layers.Reshape((-1, 32))(x)

    # One lightweight Transformer encoder
    x = lightweight_transformer(x)

    # Global aggregation
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="TerraSpectra_Light_Hybrid"
    )

    return model


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("TERRASPECTRA - LIGHTWEIGHT HYBRID TRAINING")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train = load_dataset("train")
    X_val, y_val = load_dataset("val")

    # Prepare data
    X_train = prepare_data(X_train)
    X_val = prepare_data(X_val)

    print("\nAfter preparation:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")

    # Build model
    model = build_light_hybrid(
        X_train.shape[1:],
        NUM_CLASSES
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nMODEL ARCHITECTURE")
    model.summary()

    # Paths
    best_model_path = MODELS_DIR / "best_light_hybrid.keras"
    final_model_path = MODELS_DIR / "final_light_hybrid.keras"
    history_path = RESULTS_DIR / "training_history_light_hybrid.csv"

    # Callbacks
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

    # Train
    print("\nStarting lightweight hybrid training...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=training_callbacks,
        verbose=1
    )

    # Save final model
    model.save(final_model_path)

    # Save history
    pd.DataFrame(history.history).to_csv(
        history_path,
        index=False
    )

    # Results
    best_train_accuracy = max(history.history["accuracy"])
    best_val_accuracy = max(history.history["val_accuracy"])

    best_train_epoch = (
        np.argmax(history.history["accuracy"]) + 1
    )

    best_val_epoch = (
        np.argmax(history.history["val_accuracy"]) + 1
    )

    print("\n" + "=" * 60)
    print("LIGHTWEIGHT HYBRID TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training history: {history_path}")

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

    print("\nImproved 3D-CNN benchmark: 0.5250")

    if best_val_accuracy > 0.5250:
        print("✓ Lightweight hybrid outperformed Improved 3D-CNN")
    else:
        print("⚠ Improved 3D-CNN remains the best model")

    print("=" * 60)


if __name__ == "__main__":
    main()