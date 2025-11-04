"""
Musical Components for Enhanced Model

Includes:
1. Multi-scale temporal encoder (sees patterns at different time scales)
2. Musical transition model (learns plausible note-to-note transitions)
3. Improved onset/offset loss (temporal consistency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleTemporalEncoder(nn.Module):
    """
    Process audio at multiple time scales simultaneously.
    
    This helps the model see both:
    - Fine details (sharp onsets, quick transitions)
    - Coarse structure (phrase boundaries, musical context)
    
    Uses dilated convolutions to achieve different receptive fields.
    """
    
    def __init__(self, hidden_size: int = 256, num_scales: int = 4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_scales = num_scales
        
        # Create parallel convolutional paths with different dilations
        self.scale_convs = nn.ModuleList()
        
        dilation_rates = [1, 2, 4, 8][:num_scales]
        
        for i, dilation in enumerate(dilation_rates):
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_size,
                        hidden_size // num_scales,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation
                    ),
                    nn.BatchNorm1d(hidden_size // num_scales),
                    nn.ReLU()
                )
            )
        
        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, hidden_size)
        
        Returns:
            output: (batch, time, hidden_size)
        """
        batch, time, hidden = x.shape
        
        # Transpose for Conv1d: (batch, hidden, time)
        x_t = x.permute(0, 2, 1)
        
        # Process at each scale
        scale_outputs = []
        for conv in self.scale_convs:
            scale_out = conv(x_t)
            scale_outputs.append(scale_out)
        
        # Concatenate all scales
        multi_scale = torch.cat(scale_outputs, dim=1)  # (batch, hidden, time)
        
        # Combine with 1x1 conv
        combined = self.combine(multi_scale)
        combined = self.dropout(combined)
        
        # Back to (batch, time, hidden)
        output = combined.permute(0, 2, 1)
        
        # Residual connection
        return output + x


class MusicalTransitionModel(nn.Module):
    """
    Learn plausible note-to-note transitions.
    
    This teaches the model musical rules:
    - Small steps are common (C -> D)
    - Large jumps are rare (C -> G#)
    - Octave jumps are okay (C4 -> C5)
    - Some intervals are more common (perfect 5th, 4th)
    """
    
    def __init__(self, num_notes: int = 88, smoothing_strength: float = 0.3):
        super().__init__()
        
        self.num_notes = num_notes
        self.smoothing_strength = smoothing_strength
        
        # Learnable transition matrix (note_i -> note_j)
        self.transition_logits = nn.Parameter(
            torch.zeros(num_notes, num_notes)
        )
        
        # Initialize with musical priors
        self._init_musical_priors()
    
    def _init_musical_priors(self):
        """Initialize transition matrix with musical knowledge."""
        num_notes = self.num_notes
        
        with torch.no_grad():
            for i in range(num_notes):
                for j in range(num_notes):
                    interval = abs(i - j)
                    
                    # Same note (sustain) - very common
                    if interval == 0:
                        self.transition_logits[i, j] = 3.0
                    
                    # Steps (1-2 semitones) - very common
                    elif interval <= 2:
                        self.transition_logits[i, j] = 2.0
                    
                    # Small leaps (3-5 semitones) - common
                    elif interval <= 5:
                        self.transition_logits[i, j] = 1.0
                    
                    # Perfect 5th (7 semitones) - common in melodies
                    elif interval == 7:
                        self.transition_logits[i, j] = 1.5
                    
                    # Octaves (12 semitones) - acceptable
                    elif interval == 12:
                        self.transition_logits[i, j] = 1.0
                    
                    # Large leaps - rare
                    else:
                        self.transition_logits[i, j] = -1.0
    
    def forward(self, frame_logits: torch.Tensor) -> torch.Tensor:
        """
        Smooth frame predictions using learned musical transitions.
        
        Args:
            frame_logits: (batch, time, num_notes) - raw predictions
        
        Returns:
            smoothed: (batch, time, num_notes) - musically smoothed
        """
        batch, time, notes = frame_logits.shape
        
        if time <= 1:
            return frame_logits
        
        # Get transition probabilities
        transition_probs = torch.softmax(self.transition_logits, dim=1)
        
        # Initialize output
        smoothed = frame_logits.clone()
        
        # Apply temporal smoothing (forward pass)
        for t in range(1, time):
            # Previous frame probabilities
            prev_probs = torch.softmax(smoothed[:, t-1, :], dim=-1)
            
            # Expected next note distribution based on transitions
            # (batch, notes) @ (notes, notes) = (batch, notes)
            expected_next = torch.matmul(prev_probs, transition_probs)
            
            # Blend with current prediction
            # Higher smoothing_strength = more musical constraint
            current_probs = torch.softmax(frame_logits[:, t, :], dim=-1)
            blended_probs = (
                self.smoothing_strength * expected_next +
                (1 - self.smoothing_strength) * current_probs
            )
            
            # Convert back to logits
            smoothed[:, t, :] = torch.log(blended_probs + 1e-8)
        
        return smoothed


class ImprovedOnsetOffsetLoss(nn.Module):
    """
    Enhanced loss for onset/offset detection with temporal consistency.

    Improvements over basic BCE:
    1. Temporal consistency - onsets and offsets should be paired
    2. Minimum duration - notes can't be too short
    3. Balanced weighting - adjusted for class imbalance
    4. Dynamic gradient balancing - prevents onset head death
    """

    def __init__(
        self,
        onset_weight: float = 5.0,  # Reduced with dual-LSTM (was 20.0, caused 100x effective weight)
        offset_weight: float = 3.0,  # Reduced proportionally (was 10.0)
        consistency_weight: float = 0.1,  # Reduced - was too harsh
        pairing_weight: float = 0.05,  # Reduced - was too harsh
        min_duration_frames: int = 3,
        sparsity_weight: float = 0.0,  # DISABLED temporarily - was pushing already-collapsed preds further down
        sparsity_target: float = 0.0336,  # Measured dataset onset density
        device: str = 'cpu'
    ):
        super().__init__()

        self.onset_weight = onset_weight
        self.offset_weight = offset_weight
        self.consistency_weight = consistency_weight
        self.pairing_weight = pairing_weight
        self.min_duration_frames = min_duration_frames
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

        # POS/NEG AVERAGED BCE - minimal coefficient to prevent gradient explosion
        # Even pos_coeff=4 caused 200-350x gradient spikes (1 positive × 4 × 5 internal = 20×)
        # Using pos_coeff=1.5 + tight clamp=5.0 + temporal_loss_scale=0.1
        self.pos_coeff = 1.5  # Minimal multiplication (was 4, still too high)
        self.max_pos_loss = 5.0  # Tight clamp on per-positive loss (was 20, still too high)

    def _posneg_bce(self, logits: torch.Tensor, targets: torch.Tensor, pos_coeff: float = 1.5) -> torch.Tensor:
        """
        Compute pos/neg averaged BCE to prevent count imbalance.

        Standard BCE with pos_weight still suffers from count imbalance because
        reduction='mean' divides by total count. With 3998 negatives and 2 positives,
        the negative gradient dominates even with pos_weight=50.

        Solution: Average positives and negatives separately:
            pos_loss = BCE[positive samples only].mean()  # Average over ~2 samples
            neg_loss = BCE[negative samples only].mean()  # Average over ~3998 samples
            total = pos_coeff * pos_loss + neg_loss

        This completely decouples the loss from class counts.

        Args:
            logits: (batch, time) predicted logits
            targets: (batch, time) binary targets
            pos_coeff: weight for positive class loss (default 30.0 ≈ neg/pos ratio)

        Returns:
            Scalar loss
        """
        import torch.nn.functional as F

        # Flatten
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Compute element-wise BCE (no reduction yet)
        loss_elem = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='none')

        # Separate positive and negative samples
        pos_mask = targets_flat == 1
        neg_mask = ~pos_mask

        # Average pos and neg separately
        if pos_mask.any():
            # Clamp individual positive losses to prevent catastrophic spikes (1-2 onsets → huge gradient)
            pos_losses = loss_elem[pos_mask]
            pos_losses_clamped = torch.clamp(pos_losses, max=self.max_pos_loss)
            pos_loss = pos_losses_clamped.mean()
        else:
            pos_loss = torch.tensor(0.0, device=loss_elem.device)

        if neg_mask.any():
            neg_loss = loss_elem[neg_mask].mean()
        else:
            neg_loss = torch.tensor(0.0, device=loss_elem.device)

        # Combine with coefficient
        return pos_coeff * pos_loss + neg_loss

    def forward(
        self,
        pred_onsets: torch.Tensor,
        pred_offsets: torch.Tensor,
        target_onsets: torch.Tensor,
        target_offsets: torch.Tensor
    ) -> dict:
        """
        Calculate all onset/offset losses.
        
        Args:
            pred_onsets: (batch, time) or (batch, time, 1)
            pred_offsets: (batch, time) or (batch, time, 1)
            target_onsets: (batch, time)
            target_offsets: (batch, time)
        
        Returns:
            dict with individual and total losses
        """
        # Squeeze if needed
        if pred_onsets.dim() == 3:
            pred_onsets = pred_onsets.squeeze(-1)
        if pred_offsets.dim() == 3:
            pred_offsets = pred_offsets.squeeze(-1)
        
        # 1. Primary loss: Pos/Neg Averaged BCE with gentle coefficient
        # Standard BCE caused complete onset suppression (mean pred 0.3%)
        # pos_coeff=4 + per-positive clamp prevents both collapse and gradient spikes
        onset_loss = self._posneg_bce(pred_onsets, target_onsets, pos_coeff=self.pos_coeff)
        offset_loss = self._posneg_bce(pred_offsets, target_offsets, pos_coeff=self.pos_coeff)
        
        # 2. Sparsity regularization (Phase 2: prevent predict-everywhere-high)
        # Encourage mean onset prediction to match dataset density (~3.36%)
        # This prevents the model from predicting high everywhere
        onset_probs = torch.sigmoid(pred_onsets)
        offset_probs = torch.sigmoid(pred_offsets)

        mean_onset_pred = onset_probs.mean()
        sparsity_loss = self.sparsity_weight * (mean_onset_pred - self.sparsity_target).pow(2)

        # 3. Temporal consistency loss (IMPROVED)
        # Only penalize STRONG onsets immediately followed by STRONG offsets
        # This prevents suppressing all onset predictions
        consistency_loss = torch.tensor(0.0, device=pred_onsets.device)

        if pred_onsets.shape[1] > self.min_duration_frames:
            # Only penalize if BOTH onset and offset are confident (>0.5)
            # This allows weak predictions to exist without penalty
            for t in range(self.min_duration_frames, pred_onsets.shape[1]):
                # Check for STRONG recent onsets (threshold at 0.5)
                recent_onsets = onset_probs[:, t-self.min_duration_frames:t].max(dim=1)[0]
                current_offsets = offset_probs[:, t]

                # Only penalize if BOTH are strong predictions
                strong_onset_mask = (recent_onsets > 0.5).float()
                strong_offset_mask = (current_offsets > 0.5).float()

                # Penalize only the confident violations
                violation = recent_onsets * current_offsets * strong_onset_mask * strong_offset_mask
                consistency_loss += violation.mean()

            consistency_loss /= (pred_onsets.shape[1] - self.min_duration_frames)
        
        # 3. Onset-offset pairing loss (IMPROVED)
        # Total count of onsets should roughly equal total count of offsets
        # Use L1 loss instead of MSE - less harsh on early training
        total_onsets = onset_probs.sum(dim=1)  # (batch,)
        total_offsets = offset_probs.sum(dim=1)  # (batch,)

        # L1 loss is more forgiving than MSE for large differences
        pairing_loss = F.l1_loss(total_onsets, total_offsets)
        
        # Total loss (including sparsity regularization)
        total_loss = (
            self.onset_weight * onset_loss +
            self.offset_weight * offset_loss +
            self.consistency_weight * consistency_loss +
            self.pairing_weight * pairing_loss +
            sparsity_loss  # Phase 2: prevent predict-everywhere-high
        )

        return {
            'onset': onset_loss,
            'offset': offset_loss,
            'consistency': consistency_loss,
            'pairing': pairing_loss,
            'sparsity': sparsity_loss,  # Phase 2: track sparsity regularization
            'mean_onset_pred': mean_onset_pred,  # For monitoring
            'total': total_loss
        }


def test_musical_components():
    """Test the musical components."""
    print("\nTesting Musical Components...")
    
    batch_size = 2
    time_steps = 125
    hidden_size = 256
    num_notes = 88
    
    # Test Multi-scale Encoder
    print("\n1. Testing MultiScaleTemporalEncoder...")
    encoder = MultiScaleTemporalEncoder(hidden_size=hidden_size)
    x = torch.randn(batch_size, time_steps, hidden_size)
    output = encoder(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == x.shape, "Shape mismatch!"
    print("  ✅ Multi-scale encoder working")
    
    # Test Musical Transition Model
    print("\n2. Testing MusicalTransitionModel...")
    transition_model = MusicalTransitionModel(num_notes=num_notes)
    frame_logits = torch.randn(batch_size, time_steps, num_notes)
    smoothed = transition_model(frame_logits)
    print(f"  Input shape: {frame_logits.shape}")
    print(f"  Output shape: {smoothed.shape}")
    print("  ✅ Transition model working")
    
    # Test Improved Loss
    print("\n3. Testing ImprovedOnsetOffsetLoss...")
    loss_fn = ImprovedOnsetOffsetLoss()
    
    pred_onsets = torch.randn(batch_size, time_steps)
    pred_offsets = torch.randn(batch_size, time_steps)
    target_onsets = torch.zeros(batch_size, time_steps)
    target_offsets = torch.zeros(batch_size, time_steps)
    
    # Add some positive examples
    target_onsets[:, [10, 50, 90]] = 1.0
    target_offsets[:, [20, 60, 100]] = 1.0
    
    losses = loss_fn(pred_onsets, pred_offsets, target_onsets, target_offsets)
    
    print(f"  Losses:")
    for key, value in losses.items():
        print(f"    {key}: {value.item():.4f}")
    print("  ✅ Improved loss working")
    
    print("\n✅ All musical components working!")


if __name__ == '__main__':
    test_musical_components()
