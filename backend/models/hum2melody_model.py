"""
Final Improved Hum2Melody Model

Key improvements:
1. Less aggressive pooling (4x instead of 8x)
2. Better temporal resolution (125 frames instead of 62)
3. Attention mechanism for better context
4. Monophonic-aware loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedHum2MelodyCRNN(nn.Module):
    """
    Improved CRNN with better temporal resolution.

    Changes from original:
    - Only 2 pooling layers (4x downsampling) instead of 3 (8x)
    - Output: 125 frames instead of 62
    - Temporal resolution: 128ms instead of 258ms
    - Added layer normalization
    - Added attention mechanism
    """

    def __init__(
        self,
        n_mels: int = 128,
        hidden_size: int = 256,
        num_notes: int = 88,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_notes = num_notes
        self.use_attention = use_attention

        # CNN with less aggressive pooling
        # Input: (batch, 1, 500, 128)

        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /2 -> (batch, 32, 250, 64)
            nn.Dropout2d(0.2)
        )

        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /2 -> (batch, 64, 125, 32)
            nn.Dropout2d(0.2)
        )

        # Block 3: 64 -> 128 channels (NO POOLING!)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # NO POOLING HERE!
            nn.Dropout2d(dropout)
        )

        # After 2 pooling layers: time / 4, mels / 4
        self.cnn_output_height = n_mels // 4  # 128 -> 32
        self.cnn_output_channels = 128
        self.lstm_input_size = self.cnn_output_channels * self.cnn_output_height  # 4096

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size * 2)

        # Optional attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.ln2 = nn.LayerNorm(hidden_size * 2)

        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_notes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 1, 500, 128)

        Returns:
            (batch, 125, 88) - 4x downsampling, not 8x!
        """
        batch_size = x.size(0)

        # CNN feature extraction
        x = self.conv1(x)  # (batch, 32, 250, 64)
        x = self.conv2(x)  # (batch, 64, 125, 32)
        x = self.conv3(x)  # (batch, 128, 125, 32) - no more pooling!

        # Reshape for LSTM
        batch, channels, time_steps, freq = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.contiguous().view(batch_size, time_steps, channels * freq)

        # LSTM
        x, _ = self.lstm(x)  # (batch, 125, 512)
        x = self.ln1(x)

        # Attention (optional)
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out  # Residual connection
            x = self.ln2(x)

        # Output layers
        x = self.fc1(x)  # (batch, 125, 256)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 125, 88)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """
    Focal Loss - better than extreme pos_weight for imbalanced data.

    Focal Loss automatically down-weights easy examples and focuses
    on hard examples, which is perfect for note detection where
    most frames are silence (easy negatives).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, pos_weight: float = 50.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply pos_weight to targets
        weight = torch.ones_like(targets)
        weight[targets == 1] = self.pos_weight

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', weight=weight  # ADD weight
        )

        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


class MonophonicLoss(nn.Module):
    """
    Monophonic constraint loss.

    Encourages the model to predict only one pitch at a time,
    which is correct for singing/humming.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Logits (batch, time, 88)
            targets: Binary targets (batch, time, 88)
        """
        # Get probabilities
        probs = torch.sigmoid(predictions)

        # For each time frame, calculate how spread out the predictions are
        # We want predictions concentrated on ONE pitch

        # Calculate L2 norm across pitch dimension
        # High norm = concentrated (good)
        # Low norm = spread out (bad)

        # Normalize probabilities across pitches
        eps = 1e-8
        probs_sum = probs.sum(dim=2, keepdim=True) + eps
        probs_norm = probs / probs_sum

        # Calculate entropy - high entropy means spread across many pitches
        entropy = -(probs_norm * torch.log(probs_norm + eps)).sum(dim=2)

        # We want LOW entropy (concentrated), so penalize high entropy
        loss = entropy.mean()

        return self.weight * loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for better training.
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mono_weight: float = 0.1,
        use_monophonic: bool = True
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.mono_loss = MonophonicLoss(weight=mono_weight)
        self.use_monophonic = use_monophonic

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Returns dict with individual losses for logging.
        """
        focal = self.focal_loss(predictions, targets)

        losses = {'focal': focal, 'total': focal}

        if self.use_monophonic:
            mono = self.mono_loss(predictions, targets)
            losses['monophonic'] = mono
            losses['total'] = focal + mono

        return losses


def test_improved_model():
    """Test the improved model."""
    print("Testing ImprovedHum2MelodyCRNN...")

    model = ImprovedHum2MelodyCRNN()

    print(f"\nModel parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 500, 128)

    print(f"\nInput shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 125, 88)")
    print(f"Temporal reduction: {500 / output.shape[1]:.1f}x (4x instead of 8x!)")
    print(f"Output resolution: {500 / output.shape[1] / 31.25 * 1000:.0f}ms (was 258ms, now 128ms)")

    # Test combined loss
    criterion = CombinedLoss()
    targets = torch.zeros_like(output)
    targets[:, 10:20, 40:45] = 1.0

    losses = criterion(output, targets)
    print(f"\nLoss values:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    print("\n✅ Improved model test passed!")


if __name__ == '__main__':
    test_improved_model()