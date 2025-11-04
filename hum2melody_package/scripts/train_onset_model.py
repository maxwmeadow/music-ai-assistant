"""
Train Onset/Offset Model

Trains a standalone onset/offset detection model.

This model is designed to be combined with the pitch detection model
into a single deployable artifact.

Key differences from pitch training:
- Uses MelodyDataset (CQT-only, 88 bins)
- Only trains onset/offset heads (no frame/f0)
- Saves checkpoints with preprocessing metadata for wrapper compatibility
- Monitors onset/offset F1 specifically

Usage:
    python scripts/train_onset_model.py \\
        --labels dataset/combined_manifest.json \\
        --n-bins 88 \\
        --hidden-size 128 \\
        --batch-size 32 \\
        --lr 1e-4 \\
        --epochs 50 \\
        --checkpoint-dir checkpoints_onset
"""

import sys
from pathlib import Path
import os
import json
import argparse
from typing import Dict, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Import our modules
from models.onset_model import OnsetOffsetModel
from models.musical_components import ImprovedOnsetOffsetLoss
from data.melody_dataset import EnhancedMelodyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Onset/Offset Model")

    # Data
    parser.add_argument("--labels", type=str, required=True, help="Path to labels manifest")
    parser.add_argument("--train-split", type=float, default=0.9)

    # Model architecture
    parser.add_argument("--n-bins", type=int, default=88, help="Number of CQT bins")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-amp", action='store_true', help="Use mixed precision")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--onset-weight", type=float, default=5.0)
    parser.add_argument("--offset-weight", type=float, default=3.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--pairing-weight", type=float, default=0.05)
    parser.add_argument("--sparsity-weight", type=float, default=0.0)

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_onset")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def save_checkpoint_with_metadata(
    model, optimizer, scheduler, epoch, metrics, dataset, checkpoint_path
):
    """Save checkpoint with complete preprocessing metadata for wrapper compatibility."""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': metrics,

        # CRITICAL: Preprocessing metadata (must match pitch model)
        'preprocessing': {
            'sr': dataset.sample_rate,
            'hop_length': dataset.hop_length,
            'n_bins': dataset.n_bins,
            'target_frames': dataset.target_frames,
            'fmin': dataset.fmin,
            'bins_per_octave': dataset.bins_per_octave,
            'min_midi': dataset.min_midi,
            'max_midi': dataset.max_midi,

            # CQT normalization
            'cqt_ref_db': -80,  # From dataset._load_cqt
            'cqt_range_db': 80,

            # CNN downsampling (must match pitch model)
            'cnn_downsample_factor': 4,  # 2 pooling layers
            'expected_output_frames': 125,  # 500 / 4
        },

        # Architecture metadata
        'architecture': {
            'model_class': model.__class__.__name__,
            'n_bins': model.n_bins,
            'input_channels': model.n_bins,  # Onset model uses CQT only
            'hidden_size': model.hidden_size,
            'num_notes': 88,  # Not used, but kept for consistency
        }
    }

    torch.save(checkpoint, checkpoint_path)


def interpolate_target(target, target_name, src_frames, dst_frames, device):
    """
    Resample target from src_frames -> dst_frames.

    CRITICAL: Different target types need different interpolation modes!
    - Onset/Offset (SPARSE binary): Use 'nearest' to preserve rare events
    """
    t = target.to(device).float()

    # Normalize shape to (batch, channels, time) for interpolate
    if t.dim() == 2:
        t = t.unsqueeze(-1)

    batch, s_frames, channels = t.shape
    t = t.permute(0, 2, 1)  # (batch, channels, time)
    t = t.unsqueeze(-1)  # (batch, channels, time, 1)

    # SPARSE binary: use nearest to preserve rare events
    interpolation_mode = 'nearest'
    is_binary = True

    interpolated = F.interpolate(
        t, size=(dst_frames, 1), mode=interpolation_mode,
        align_corners=None
    )
    interpolated = interpolated.squeeze(-1)  # (batch, channels, dst_frames)
    interpolated = interpolated.permute(0, 2, 1)  # (batch, dst_frames, channels)

    if is_binary:
        # Apply threshold to ensure binary values
        interpolated = (interpolated > 0.5).float()
        if channels == 1:
            interpolated = interpolated.squeeze(-1)

    return interpolated


def calculate_onset_metrics(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculate onset/offset F1 metrics."""
    out = {}

    def _binary_metric(pred_logits, target, name):
        pt = pred_logits.detach().cpu()
        tt = target.detach().cpu()

        if pt.dim() == 3 and pt.shape[-1] == 1:
            pt = pt.squeeze(-1)
        if tt.dim() == 3 and tt.shape[-1] == 1:
            tt = tt.squeeze(-1)

        # Convert logits to probabilities
        prob = torch.sigmoid(pt)

        # Use low threshold for onset/offset (sparse events)
        threshold = 0.15
        pred_bin = (prob > threshold).float()

        p_flat = pred_bin.numpy().reshape(-1)
        t_flat = (tt.numpy().reshape(-1) > 0.5).astype(float)

        tp = float(((p_flat == 1) & (t_flat == 1)).sum())
        fp_cnt = float(((p_flat == 1) & (t_flat == 0)).sum())
        fn = float(((p_flat == 0) & (t_flat == 1)).sum())

        prec = tp / (tp + fp_cnt + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        return float(f1), float(prec), float(rec)

    onset_f1, onset_prec, onset_rec = _binary_metric(preds['onset'], targets['onset'], 'onset')
    offset_f1, offset_prec, offset_rec = _binary_metric(preds['offset'], targets['offset'], 'offset')

    out['onset_f1'] = onset_f1
    out['onset_precision'] = onset_prec
    out['onset_recall'] = onset_rec
    out['offset_f1'] = offset_f1
    out['offset_precision'] = offset_prec
    out['offset_recall'] = offset_rec

    return out


def train_epoch(
    model, train_loader, criterion, optimizer, scheduler,
    device, epoch, use_amp=False, grad_clip=1.0
):
    """Train for one epoch."""
    print(f"\n{'='*60}")
    print(f"TRAINING EPOCH {epoch + 1}")
    print(f"{'='*60}")

    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_losses = {'onset': 0.0, 'offset': 0.0, 'consistency': 0.0, 'pairing': 0.0, 'total': 0.0}
    all_metrics = {
        'onset_f1': [],
        'offset_f1': [],
    }

    metrics = {'onset_f1': 0.0, 'offset_f1': 0.0}

    train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")

    for batch_idx, (data, targets) in enumerate(train_iter):
        data = data.to(device)
        targets = {k: v.to(device).float() for k, v in targets.items()}

        optimizer.zero_grad()

        # Forward
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(data)

            # Resample targets to match model output
            output_frames = outputs['onset'].shape[1]
            target_frames = targets['onset'].shape[1]

            if output_frames != target_frames:
                targets['onset'] = interpolate_target(
                    targets['onset'], 'onset', target_frames, output_frames, device
                )
                targets['offset'] = interpolate_target(
                    targets['offset'], 'offset', target_frames, output_frames, device
                )

            losses = criterion(
                outputs['onset'],
                outputs['offset'],
                targets['onset'],
                targets['offset']
            )
            loss = losses['total']

        # Backward with gradient clipping
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Accumulate losses
        for k in ['onset', 'offset', 'consistency', 'pairing', 'total']:
            if k in losses:
                val = losses[k].item() if isinstance(losses[k], torch.Tensor) else float(losses[k])
                total_losses[k] += val

        # Calculate metrics periodically
        if batch_idx % 5 == 0:
            batch_metrics = calculate_onset_metrics(outputs, targets)
            all_metrics['onset_f1'].append(batch_metrics['onset_f1'])
            all_metrics['offset_f1'].append(batch_metrics['offset_f1'])

            metrics = {
                'onset_f1': batch_metrics['onset_f1'],
                'offset_f1': batch_metrics['offset_f1'],
            }

        # Update progress bar
        train_iter.set_postfix({
            'loss': f"{loss.item():.4f}",
            'onset_f1': f"{metrics['onset_f1']:.3f}",
            'offset_f1': f"{metrics['offset_f1']:.3f}"
        })

    # Finalize epoch metrics
    num_batches = len(train_loader)
    avg_losses = {k: (total_losses[k] / max(1, num_batches)) for k in total_losses}

    epoch_metrics = {
        'total': avg_losses['total'],
        'onset': avg_losses['onset'],
        'offset': avg_losses['offset'],
        'consistency': avg_losses['consistency'],
        'pairing': avg_losses['pairing'],
        'onset_f1': float(np.mean(all_metrics['onset_f1'])) if all_metrics['onset_f1'] else 0.0,
        'offset_f1': float(np.mean(all_metrics['offset_f1'])) if all_metrics['offset_f1'] else 0.0,
    }

    print(f"\nEpoch {epoch + 1} Training Complete:")
    print(f"  Loss: {epoch_metrics['total']:.4f}")
    print(f"  Onset F1: {epoch_metrics['onset_f1']:.4f}")
    print(f"  Offset F1: {epoch_metrics['offset_f1']:.4f}")

    return epoch_metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch."""
    print(f"\n{'='*60}")
    print(f"VALIDATING EPOCH {epoch + 1}")
    print(f"{'='*60}")

    model.eval()

    total_losses = {'onset': 0.0, 'offset': 0.0, 'consistency': 0.0, 'pairing': 0.0, 'total': 0.0}
    all_metrics = {
        'onset_f1': [],
        'offset_f1': [],
    }

    val_iter = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}")

    with torch.no_grad():
        for data, targets in val_iter:
            data = data.to(device)
            targets = {k: v.to(device).float() for k, v in targets.items()}

            predictions = model(data)

            # Resample targets
            output_frames = predictions['onset'].shape[1]
            target_frames = targets['onset'].shape[1]

            if output_frames != target_frames:
                targets['onset'] = interpolate_target(
                    targets['onset'], 'onset', target_frames, output_frames, device
                )
                targets['offset'] = interpolate_target(
                    targets['offset'], 'offset', target_frames, output_frames, device
                )

            losses = criterion(
                predictions['onset'],
                predictions['offset'],
                targets['onset'],
                targets['offset']
            )

            for key, value in losses.items():
                if key in total_losses:
                    total_losses[key] += value.item() if isinstance(value, torch.Tensor) else float(value)

            metrics = calculate_onset_metrics(predictions, targets)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

    # Finalize
    result = {k: v / len(val_loader) for k, v in total_losses.items()}
    for key in all_metrics:
        result[key] = np.mean(all_metrics[key])

    print(f"\nEpoch {epoch + 1} Validation Complete:")
    print(f"  Loss: {result['total']:.4f}")
    print(f"  Onset F1: {result['onset_f1']:.4f}")
    print(f"  Offset F1: {result['offset_f1']:.4f}")

    return result


def main():
    args = parse_args()
    torch.manual_seed(42)

    # Device setup
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n{'='*70}")
    print("ONSET/OFFSET MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    if args.use_amp and device.type != 'cuda':
        print("Warning: AMP only works on CUDA, disabling")
        args.use_amp = False

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max LR: {args.max_lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  CQT bins: {args.n_bins}")
    print(f"{'='*70}\n")

    # Create dataset (CQT-only, no extra features)
    # Use EnhancedMelodyDataset to get offset targets, but disable all extra features
    print("Creating dataset...")
    dataset = EnhancedMelodyDataset(
        labels_path=args.labels,
        n_bins=args.n_bins,
        augment=True,
        spec_augment=True,
        use_onset_features=False,
        use_musical_context=False,
        use_pretrained=False
    )

    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {train_size}, Val: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )

    # Create model
    print("\nCreating model...")
    model = OnsetOffsetModel(
        n_bins=args.n_bins,
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    model = model.to(device)
    print(f"✅ Model created: {model.count_parameters():,} parameters")

    # Create loss criterion
    print("\nCreating loss criterion...")
    criterion = ImprovedOnsetOffsetLoss(
        onset_weight=args.onset_weight,
        offset_weight=args.offset_weight,
        consistency_weight=args.consistency_weight,
        pairing_weight=args.pairing_weight,
        sparsity_weight=args.sparsity_weight,
        device=str(device)
    )
    print("✅ Loss criterion created")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    print(f"✅ Optimizer and scheduler created")
    print(f"   Steps per epoch: {len(train_loader)}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"✅ Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_onset_f1': [],
        'val_onset_f1': [],
        'train_offset_f1': [],
        'val_offset_f1': [],
        'lr': []
    }

    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")

    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip
        )

        # Validate
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['train_onset_f1'].append(train_metrics['onset_f1'])
        history['val_onset_f1'].append(val_metrics['onset_f1'])
        history['train_offset_f1'].append(train_metrics['offset_f1'])
        history['val_offset_f1'].append(val_metrics['offset_f1'])
        history['lr'].append(current_lr)

        # Combined F1 for early stopping
        combined_f1 = (val_metrics['onset_f1'] + val_metrics['offset_f1']) / 2

        # Save checkpoint with metadata
        if combined_f1 > best_f1:
            best_f1 = combined_f1
            save_checkpoint_with_metadata(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                dataset=dataset,
                checkpoint_path=checkpoint_dir / "best_model.pth"
            )
            print(f"\n✅ New best F1: {best_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save epoch checkpoint
        save_checkpoint_with_metadata(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics,
            dataset=dataset,
            checkpoint_path=checkpoint_dir / f"epoch_{epoch + 1}.pth"
        )

        # Save history
        with open(checkpoint_dir / "train_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping check
        if epochs_no_improve >= args.patience:
            print(f"\n⚠️ Early stopping: no improvement for {args.patience} epochs")
            break

    # Training complete
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best combined F1: {best_f1:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"History saved to: {checkpoint_dir / 'train_history.json'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
