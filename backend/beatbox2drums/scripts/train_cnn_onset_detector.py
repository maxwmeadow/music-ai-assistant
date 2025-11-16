#!/usr/bin/env python3
"""
Train CNN-based onset detector for beatbox audio.

Uses prepared spectrogram windows with onset/no-onset labels.
"""

import sys
import json
import numpy as np
from pathlib import Path
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def build_onset_cnn(input_shape=(80, 12, 1)):
    """
    Build CNN for onset detection.

    Adapted from DrumOnsetDetectionCNN architecture.

    Args:
        input_shape: (n_mels, window_frames, channels)

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input
        layers.InputLayer(input_shape=input_shape),

        # Block 1
        layers.Conv2D(10, (3, 7), activation='relu', padding='same', strides=1, name='block1_conv'),
        layers.MaxPooling2D((3, 1), strides=(2, 2), padding='same', name='block1_pool'),
        layers.BatchNormalization(name='block1_norm'),

        # Block 2
        layers.Conv2D(20, (3, 3), activation='relu', padding='same', strides=1, name='block2_conv'),
        layers.MaxPooling2D((3, 1), strides=(2, 2), padding='same', name='block2_pool'),
        layers.BatchNormalization(name='block2_norm'),

        # Flatten
        layers.Flatten(name='flatten'),

        # Dense layers
        layers.Dense(256, activation='relu', name='dense'),
        layers.BatchNormalization(name='dense_norm'),
        layers.Dropout(0.5, name='dropout'),

        # Output: 2 classes (no-onset, onset)
        layers.Dense(2, activation='softmax', name='output')
    ])

    model.summary()
    return model


def load_data(data_dir: Path, split: str):
    """
    Load prepared CNN onset detection data.

    Args:
        data_dir: Directory containing prepared data
        split: 'train', 'val', or 'test'

    Returns:
        windows, labels
    """
    windows_file = data_dir / f'{split}_windows.npy'
    labels_file = data_dir / f'{split}_labels.npy'

    if not windows_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Data files not found for split '{split}' in {data_dir}")

    windows = np.load(windows_file)
    labels = np.load(labels_file)

    # Convert labels to one-hot
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=2)

    return windows, labels_onehot


def train_onset_detector(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 15,
    use_class_weights: bool = True
):
    """
    Train CNN onset detector.

    Args:
        data_dir: Directory with prepared training data
        output_dir: Directory to save trained model
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        patience: Early stopping patience
        use_class_weights: Balance class weights
    """
    print(f"\n{'='*70}")
    print("CNN Onset Detector Training")
    print(f"{'='*70}\n")

    # Load data
    print("Loading data...")
    train_windows, train_labels = load_data(data_dir, 'train')
    val_windows, val_labels = load_data(data_dir, 'val')

    print(f"Training windows: {train_windows.shape}")
    print(f"Validation windows: {val_windows.shape}")
    print()

    # Class distribution
    train_labels_1d = np.argmax(train_labels, axis=1)
    n_no_onset = np.sum(train_labels_1d == 0)
    n_onset = np.sum(train_labels_1d == 1)

    print(f"Training class distribution:")
    print(f"  No-onset: {n_no_onset:,} ({n_no_onset/len(train_labels_1d)*100:.1f}%)")
    print(f"  Onset: {n_onset:,} ({n_onset/len(train_labels_1d)*100:.1f}%)")
    print(f"  Ratio: 1:{n_no_onset/n_onset:.1f}")
    print()

    # Build model
    print("Building model...")
    input_shape = train_windows.shape[1:]  # (n_mels, window_frames, channels)
    model = build_onset_cnn(input_shape)
    print()

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            output_dir / 'best_onset_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Class weights
    class_weights_dict = None
    if use_class_weights:
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_labels_1d),
            y=train_labels_1d
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weights_dict}")
        print()

    # Train
    print(f"Training for up to {epochs} epochs (batch size: {batch_size})...")
    print(f"{'='*70}\n")

    history = model.fit(
        train_windows,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(val_windows, val_labels),
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # Evaluate
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}\n")

    # Validation set
    val_loss, val_acc = model.evaluate(val_windows, val_labels, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print()

    # Detailed metrics
    val_pred = model.predict(val_windows, verbose=0)
    val_pred_classes = np.argmax(val_pred, axis=1)
    val_true_classes = np.argmax(val_labels, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        val_true_classes,
        val_pred_classes,
        average='binary',
        pos_label=1
    )

    print(f"Onset Detection Metrics (on validation set):")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall: {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1 Score: {f1:.4f} ({f1*100:.1f}%)")
    print()

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(val_true_classes, val_pred_classes).ravel()
    print(f"Confusion Matrix:")
    print(f"  True Positives: {tp:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Negatives: {tn:,}")
    print()

    # Save final model
    model.save(output_dir / 'final_onset_model.h5')
    print(f"✓ Model saved to {output_dir}")
    print()

    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    }

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save final metrics
    metrics = {
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved to {output_dir / 'metrics.json'}")
    print(f"{'='*70}\n")

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN onset detector'
    )

    parser.add_argument('--data-dir', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_data',
                       help='Directory with prepared training data')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_model',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--no-class-weights', action='store_true',
                       help='Disable class weighting')

    args = parser.parse_args()

    train_onset_detector(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        use_class_weights=not args.no_class_weights
    )


if __name__ == '__main__':
    main()
