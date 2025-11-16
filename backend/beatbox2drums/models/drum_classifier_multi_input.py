#!/usr/bin/env python3
"""
Multi-Input CNN Classifier for Beatbox-to-Drums classification.

Combines mel spectrogram CNN with hand-crafted spectral features for improved
snare classification.

Based on research showing that spectral features (bandwidth, rolloff, centroid,
flatness) significantly improve discrimination of snare drums, especially
distinguishing between p-snare and k-snare types.

Input:
  - Mel spectrogram: (batch, 1, 128, 12)
  - Spectral features: (batch, 8) [bandwidth_mean/std, rolloff_mean/std,
                                    centroid_mean/std, flatness_mean/std]

Output: (batch, num_classes) class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrumClassifierCNN_MultiInput(nn.Module):
    """
    Multi-input CNN combining mel spectrogram and spectral features.

    Architecture:
        Branch 1 (Mel CNN): Conv2D(32) → Conv2D(64) → Conv2D(128) → GAP → (128,)
        Branch 2 (Features): Dense(32) → Dense(64) → (64,)
        Combined: Concat → Dense(128) → Dense(num_classes)

    This architecture leverages both:
    1. Learned features from mel spectrogram (CNN branch)
    2. Expert knowledge from spectral features (dense branch)
    """

    def __init__(self, num_classes=3, num_features=8, dropout=0.3):
        """
        Args:
            num_classes: Number of drum classes (default: 3 for kick/snare/hihat,
                         or 4 for kick/p-snare/k-snare/hihat)
            num_features: Number of spectral features (default: 8)
            dropout: Dropout rate (default: 0.3)
        """
        super(DrumClassifierCNN_MultiInput, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.dropout_rate = dropout

        # =================================================================
        # Branch 1: Mel Spectrogram CNN
        # =================================================================

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

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # =================================================================
        # Branch 2: Spectral Features Dense Network
        # =================================================================

        self.feature_fc1 = nn.Linear(num_features, 32)
        self.feature_bn1 = nn.BatchNorm1d(32)
        self.feature_dropout1 = nn.Dropout(0.3)

        self.feature_fc2 = nn.Linear(32, 64)
        self.feature_bn2 = nn.BatchNorm1d(64)
        self.feature_dropout2 = nn.Dropout(0.3)

        # =================================================================
        # Combined Classifier
        # =================================================================

        self.fc1 = nn.Linear(128 + 64, 128)  # Concatenate both branches
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.fc_dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, mel_spec, features):
        """
        Forward pass with two inputs.

        Args:
            mel_spec: Mel spectrogram tensor (batch, 1, 128, 12)
            features: Spectral features tensor (batch, num_features)

        Returns:
            Logits tensor (batch, num_classes)
        """
        # =================================================================
        # Branch 1: Process Mel Spectrogram
        # =================================================================

        # Conv Block 1: (batch, 1, 128, 12) → (batch, 32, 64, 6)
        x = self.conv1(mel_spec)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv Block 2: (batch, 32, 64, 6) → (batch, 64, 32, 3)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Conv Block 3: (batch, 64, 32, 3) → (batch, 128, 32, 3)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # Global Average Pooling: (batch, 128, 32, 3) → (batch, 128, 1, 1)
        x = self.gap(x)

        # Flatten: (batch, 128, 1, 1) → (batch, 128)
        x = x.view(x.size(0), -1)

        # =================================================================
        # Branch 2: Process Spectral Features
        # =================================================================

        # Dense Layer 1: (batch, num_features) → (batch, 32)
        f = self.feature_fc1(features)
        f = F.relu(f)
        f = self.feature_bn1(f)
        f = self.feature_dropout1(f)

        # Dense Layer 2: (batch, 32) → (batch, 64)
        f = self.feature_fc2(f)
        f = F.relu(f)
        f = self.feature_bn2(f)
        f = self.feature_dropout2(f)

        # =================================================================
        # Combine Branches
        # =================================================================

        # Concatenate: (batch, 128) + (batch, 64) → (batch, 192)
        combined = torch.cat([x, f], dim=1)

        # Final Classifier: (batch, 192) → (batch, num_classes)
        out = self.fc1(combined)
        out = F.relu(out)
        out = self.fc_bn1(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)

        # Return logits (CrossEntropyLoss applies softmax internally)
        return out

    def predict(self, mel_spec, features):
        """
        Predict class probabilities.

        Args:
            mel_spec: Mel spectrogram tensor (batch, 1, 128, 12)
            features: Spectral features tensor (batch, num_features)

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        logits = self.forward(mel_spec, features)
        probabilities = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes, probabilities

    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_multi_input(num_classes=3, num_features=8, dropout=0.3, device='cpu'):
    """
    Create and initialize the multi-input drum classifier model.

    Args:
        num_classes: Number of drum classes (default: 3)
        num_features: Number of spectral features (default: 8)
        dropout: Dropout rate (default: 0.3)
        device: Device to place model on ('cpu' or 'cuda')

    Returns:
        Initialized model on specified device
    """
    model = DrumClassifierCNN_MultiInput(
        num_classes=num_classes,
        num_features=num_features,
        dropout=dropout
    )
    model = model.to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    return model


if __name__ == '__main__':
    # Test model
    print("="*70)
    print("DrumClassifierCNN_MultiInput Architecture")
    print("="*70)
    print()

    # Create model
    model = create_model_multi_input(num_classes=3, num_features=8, dropout=0.3, device='cpu')

    # Print model
    print(model)
    print()

    # Print parameters
    total_params = model.get_num_params()
    print(f"Total parameters: {total_params:,}")
    print()

    # Test forward pass
    batch_size = 8
    test_mel_spec = torch.randn(batch_size, 1, 128, 12)
    test_features = torch.randn(batch_size, 8)

    print(f"Input shapes:")
    print(f"  Mel spectrogram: {test_mel_spec.shape}")
    print(f"  Features: {test_features.shape}")
    print()

    # Forward pass
    output = model(test_mel_spec, test_features)
    print(f"Output shape (logits): {output.shape}")

    # Predictions
    predicted_classes, probabilities = model.predict(test_mel_spec, test_features)
    print(f"Predicted classes shape: {predicted_classes.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predicted classes: {predicted_classes}")
    print(f"Probabilities (first sample): {probabilities[0]}")
    print(f"Probabilities sum: {probabilities[0].sum():.4f}")
    print()

    print("="*70)
    print("✓ Model test successful!")
    print("="*70)
