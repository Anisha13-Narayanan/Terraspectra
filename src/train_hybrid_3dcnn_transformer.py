from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers


# ==========================================================
# TERRASPECTRA - 3D-CNN + TRANSFORMER HYBRID
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


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_dataset(split_name):

    split_dir = PATCHES_DIR / split_name

    X = np.load(split_dir / f"X_{split_name}.npy")
    y = np.load(split_dir / f"y_{split_name}.npy")

    print("\n" + "=" * 60)
    print(f"LOADED {split_name.upper()} DATA")
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y


# ==========================================================
# PREPARE DATA
# ==========================================================

def prepare_data(X):
    X = X.astype(np.float32)
    X = np.expand_dims(X, axis=-1)
    return X


# ==========================================================
# CONVOLUTION BLOCK
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
# TRANSFORMER ENCODER
# ==========================================================

def transformer_encoder(
    inputs,
    projection_dim=64,
    num_heads=4,
    transformer_units=128,
    dropout_rate=0.2
):

    # Layer normalization before attention
    x1 = layers.LayerNormalization(epsilon=1e-6)(inputs)

    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim // num_heads,
        dropout=dropout_rate
    )(x1, x1)

    # Residual connection
    x2 = layers.Add()([inputs, attention_output])

    # Feed-forward network
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

    x3 = layers.Dense(
        transformer_units,
        activation="gelu"
    )(x3)

    x3 = layers.Dropout(dropout_rate)(x3)

    x3 = layers.Dense(projection_dim)(x3)

    # Residual connection
    outputs = layers.Add()([x2, x3])

    return outputs


# ==========================================================
# BUILD HYBRID MODEL
# ==========================================================

def build_hybrid_model(input_shape, num_classes):

    inputs = layers.Input(shape=input_shape)

    # ------------------------------------------------------
    # 3D-CNN FEATURE EXTRACTION
    # ------------------------------------------------------

    x = conv_block(inputs, 16, 0.10)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    x = conv_block(x, 32, 0.15)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)

    x = conv_block(x, 64, 0.20)

    # Shape approximately:
    # (batch, 8, 8, 7, 64)

    # ------------------------------------------------------
    # CONVERT CNN FEATURES TO TRANSFORMER TOKENS
    # ------------------------------------------------------

    x = layers.Reshape((-1, 64))(x)

    # Project features to Transformer dimension
    x = layers.Dense(64)(x)

    # ------------------------------------------------------
    # TRANSFORMER ENCODERS
    # ------------------------------------------------------

    x = transformer_encoder(
        x,
        projection_dim=64,
        num_heads=4,
        transformer_units=128,
        dropout_rate=0.20
    )

    x = transformer_encoder(
        x,
        projection_dim=64,
        num_heads=4,
        transformer_units=128,
        dropout_rate=0.20
    )

    # ------------------------------------------------------
    # GLOBAL TOKEN AGGREGATION
    # ------------------------------------------------------

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.GlobalAveragePooling1D()(x)

    # ------------------------------------------------------
    # CLASSIFICATION HEAD
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
        name="TerraSpectra_3DCNN_Transformer"
    )

    return model


# ==========================================================
# MAIN TRAINING
# ==========================================================

def main():

    print("=" * 60)
    print("TERRASPECTRA - 3D-CNN + TRANSFORMER HYBRID TRAINING")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load train and validation data
    X_train, y_train = load_dataset("train")
    X_val, y_val = load_dataset("val")

    # Prepare for Conv3D
    X_train = prepare_data(X_train)
    X_val = prepare_data(X_val)

    print("\nDATA AFTER PREPARATION")
    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")

    # Build model
    model = build_hybrid_model(
        input_shape=X_train.shape[1:],
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
    print("HYBRID MODEL ARCHITECTURE")
    print("=" * 60)

    model.summary()

    # ------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------

    best_model_path = (
        MODELS_DIR / "best_hybrid_3dcnn_transformer.keras"
    )

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
    # TRAIN
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING HYBRID MODEL TRAINING")
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
    # SAVE FINAL MODEL
    # ------------------------------------------------------

    final_model_path = (
        MODELS_DIR / "final_hybrid_3dcnn_transformer.keras"
    )

    model.save(final_model_path)

    # ------------------------------------------------------
    # SAVE HISTORY
    # ------------------------------------------------------

    history_df = pd.DataFrame(history.history)

    history_path = (
        RESULTS_DIR /
        "training_history_hybrid_3dcnn_transformer.csv"
    )

    history_df.to_csv(history_path, index=False)

    # ------------------------------------------------------
    # BEST RESULTS
    # ------------------------------------------------------

    best_train_accuracy = max(history.history["accuracy"])
    best_val_accuracy = max(history.history["val_accuracy"])

    best_train_epoch = (
        np.argmax(history.history["accuracy"]) + 1
    )

    best_val_epoch = (
        np.argmax(history.history["val_accuracy"]) + 1
    )

    print("\n" + "=" * 60)
    print("HYBRID MODEL TRAINING COMPLETED")
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

    print("\n3D-CNN benchmark validation accuracy: 0.5250")

    if best_val_accuracy > 0.5250:
        print("✓ Hybrid model outperformed the Improved 3D-CNN")
    else:
        print("⚠ Hybrid model did not outperform the Improved 3D-CNN")

    print("=" * 60)


if __name__ == "__main__":
    main()