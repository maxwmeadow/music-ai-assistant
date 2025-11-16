#!/usr/bin/env python3
"""
CNN Classifier for Beatbox-to-Drums classification.

3-class model: kick, snare, hihat
Input: (batch, 1, 128, 16) mel spectrograms
Output: (batch, 3) class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrumClassifierCNN(nn.Module):
    """
    CNN for drum classification from mel spectrograms.

    Architecture:
        Conv2D(32) → Conv2D(64) → Conv2D(128) → GlobalAvgPool → Dense(128) → Dense(3)

    Input shape: (batch, 1, 128, 16)
        - 1 channel (grayscale mel spectrogram)
        - 128 mel frequency bins
        - 16 time frames (~100ms window)

    Output shape: (batch, 3)
        - 3 classes: [kick, snare, hihat]
        - Softmax probabilities (sum to 1.0)
    """

    def __init__(self, num_classes=3, dropout=0.3):
        """
        Args:
            num_classes: Number of drum classes (default: 3)
            dropout: Dropout rate (default: 0.3)
        """
        super(DrumClassifierCNN, self).__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        # Conv Block 1: 1 → 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)

        # Conv Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)

        # Conv Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.4)

        # Global Average Pooling (replaces flatten)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 1, 128, 16)

        Returns:
            Tensor (batch, num_classes) with class logits
        """
        # Conv Block 1: (batch, 1, 128, 16) → (batch, 32, 64, 8)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv Block 2: (batch, 32, 64, 8) → (batch, 64, 32, 4)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Conv Block 3: (batch, 64, 32, 4) → (batch, 128, 32, 4)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # Global Average Pooling: (batch, 128, 32, 4) → (batch, 128, 1, 1)
        x = self.gap(x)

        # Flatten: (batch, 128, 1, 1) → (batch, 128)
        x = x.view(x.size(0), -1)

        # FC Layers: (batch, 128) → (batch, num_classes)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        # Return logits (CrossEntropyLoss applies softmax internally)
        return x

    def predict(self, x):
        """
        Predict class probabilities.

        Args:
            x: Input tensor (batch, 1, 128, 16)

        Returns:
            Tuple of (predicted_classes, probabilities)
            - predicted_classes: (batch,) with class indices
            - probabilities: (batch, num_classes) with softmax probabilities
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes, probabilities

    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=3, dropout=0.3, device='cpu'):
    """
    Create and initialize the drum classifier model.

    Args:
        num_classes: Number of drum classes (default: 3)
        dropout: Dropout rate (default: 0.3)
        device: Device to place model on ('cpu' or 'cuda')

    Returns:
        Initialized model on specified device
    """
    model = DrumClassifierCNN(num_classes=num_classes, dropout=dropout)
    model = model.to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    return model


if __name__ == '__main__':
    # Test model
    print("="*70)
    print("DrumClassifierCNN Architecture")
    print("="*70)
    print()

    # Create model
    model = create_model(num_classes=3, dropout=0.3, device='cpu')

    # Print model
    print(model)
    print()

    # Print parameters
    total_params = model.get_num_params()
    print(f"Total parameters: {total_params:,}")
    print()

    # Test forward pass
    batch_size = 8
    test_input = torch.randn(batch_size, 1, 128, 16)

    print(f"Input shape: {test_input.shape}")

    # Forward pass
    output = model(test_input)
    print(f"Output shape (logits): {output.shape}")

    # Predictions
    predicted_classes, probabilities = model.predict(test_input)
    print(f"Predicted classes shape: {predicted_classes.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predicted classes: {predicted_classes}")
    print(f"Probabilities (first sample): {probabilities[0]}")
    print(f"Probabilities sum: {probabilities[0].sum():.4f}")
    print()

    print("="*70)
    print("✓ Model test successful!")
    print("="*70)
