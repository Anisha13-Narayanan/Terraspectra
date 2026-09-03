from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks
regularizers = tf.keras.regularizers


# ==========================================================
# TERRASPECTRA - 3D-CNN TRAINING WITH SHARED PCA DATA
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

DATA_DIR = PROJECT_ROOT / "data" / "patches_shared_pca"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.0005
RANDOM_SEED = 42


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    print("\n" + "=" * 60)
    print("LOADING SHARED PCA DATASET")
    print("=" * 60)

    X_train = np.load(
        DATA_DIR / "train" / "X_train.npy"
    ).astype(np.float32)

    y_train = np.load(
        DATA_DIR / "train" / "y_train.npy"
    ).astype(np.int32)

    X_val = np.load(
        DATA_DIR / "val" / "X_val.npy"
    ).astype(np.float32)

    y_val = np.load(
        DATA_DIR / "val" / "y_val.npy"
    ).astype(np.int32)

    print(f"Train X: {X_train.shape}")
    print(f"Train y: {y_train.shape}")
    print(f"Val X:   {X_val.shape}")
    print(f"Val y:   {y_val.shape}")

    # Conv3D requires:
    # (samples, depth, height, width, channels)
    #
    # Current:
    # (samples, height, width, PCA_components)
    #
    # Treat PCA components as depth.

    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)

    print("\nAfter Conv3D preparation:")
    print(f"Train X: {X_train.shape}")
    print(f"Val X:   {X_val.shape}")

    return X_train, y_train, X_val, y_val


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model(input_shape):

    model = models.Sequential([
        
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv3D(
            16,
            kernel_size=(3, 3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling3D(
            pool_size=(2, 2, 2)
        ),
        layers.Dropout(0.20),

        # Block 2
        layers.Conv3D(
            32,
            kernel_size=(3, 3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling3D(
            pool_size=(2, 2, 2)
        ),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv3D(
            64,
            kernel_size=(3, 3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),

        # Reduce features
        layers.GlobalAveragePooling3D(),

        # Classification head
        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.Dropout(0.40),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("TERRASPECTRA - SHARED PCA 3D-CNN TRAINING")
    print("=" * 60)

    X_train, y_train, X_val, y_val = load_data()

    model = build_model(
        input_shape=X_train.shape[1:]
    )

    print("\nMODEL SUMMARY")
    print("=" * 60)

    model.summary()

    # ------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------

    best_model_path = (
        MODELS_DIR / "best_shared_pca_3dcnn.keras"
    )

    final_model_path = (
        MODELS_DIR / "final_shared_pca_3dcnn.keras"
    )

    history_path = (
        RESULTS_DIR /
        "training_history_shared_pca_3dcnn.csv"
    )

    callback_list = [

        callbacks.ModelCheckpoint(
            filepath=best_model_path,
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

    print("\nSTARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        verbose=1,
        shuffle=True
    )

    # Save final model
    model.save(final_model_path)

    # Save history
    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_val_epoch = (
        np.argmax(history.history["val_accuracy"]) + 1
    )

    best_train_accuracy = max(
        history.history["accuracy"]
    )

    print("\n" + "=" * 60)
    print("SHARED PCA 3D-CNN TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best model saved: {best_model_path}")
    print(f"Final model saved: {final_model_path}")
    print(f"Training history: {history_path}")

    print("\nBEST RESULTS")
    print(
        f"Best training accuracy: "
        f"{best_train_accuracy:.4f}"
    )
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f} "
        f"(Epoch {best_val_epoch})"
    )

    print("\nPrevious pipeline best validation accuracy: 0.5250")

    if best_val_accuracy > 0.5250:
        print("✓ Shared PCA pipeline performed better")
    else:
        print(
            "Shared PCA pipeline did not yet outperform "
            "the previous validation result"
        )


if __name__ == "__main__":
    main()