from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers


PROJECT_ROOT = Path(r"E:\Terraspectra")
DATA_DIR = PROJECT_ROOT / "data" / "patches_shared_pca"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0003
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_split(split_name):
    split_dir = DATA_DIR / split_name
    X = np.load(split_dir / f"X_{split_name}.npy").astype(np.float32)
    y = np.load(split_dir / f"y_{split_name}.npy").astype(np.int32)

    if X.ndim != 4 or X.shape[1:] != (32, 32, 30):
        raise ValueError(f"Unexpected {split_name} shape: {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"{split_name} X/y sample counts do not match")

    return X, y


def transformer_block(inputs, projection_dim, num_heads, transformer_units):
    attention_input = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim // num_heads,
        dropout=0.15,
    )(attention_input, attention_input)
    x = layers.Add()([inputs, attention_output])

    feed_forward_input = layers.LayerNormalization(epsilon=1e-6)(x)
    feed_forward = layers.Dense(
        transformer_units,
        activation=tf.nn.gelu,
        kernel_regularizer=regularizers.l2(1e-4),
    )(feed_forward_input)
    feed_forward = layers.Dropout(0.15)(feed_forward)
    feed_forward = layers.Dense(projection_dim)(feed_forward)
    return layers.Add()([x, feed_forward])


def build_vit(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    patch_size = 4
    projection_dim = 64

    patches = layers.Conv2D(
        projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)
    token_count = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    tokens = layers.Reshape((token_count, projection_dim))(patches)

    positions = tf.range(start=0, limit=token_count, delta=1)
    position_embedding = layers.Embedding(
        input_dim=token_count,
        output_dim=projection_dim,
    )(positions)
    x = tokens + position_embedding

    x = transformer_block(x, projection_dim, 4, 128)
    x = transformer_block(x, projection_dim, 4, 128)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="TerraSpectra_ViT")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    model = build_vit(X_train.shape[1:], NUM_CLASSES)
    model.summary()

    best_model_path = MODELS_DIR / "best_vit.keras"
    history_path = RESULTS_DIR / "training_history_vit.csv"

    training_callbacks = [
        callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
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
        callbacks=training_callbacks,
        shuffle=True,
        verbose=1,
    )

    model.save(MODELS_DIR / "final_vit.keras")
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    print("\nViT training completed")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best model: {best_model_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()