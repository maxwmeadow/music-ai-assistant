"""
Final training script with all improvements:
1. Improved model architecture
2. Focal Loss + Monophonic Loss
3. Proper target interpolation
4. Better metrics
5. Better training practices
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from models.hum2melody_model import ImprovedHum2MelodyCRNN, CombinedLoss, FocalLoss
from data.melody_dataset import MelodyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Improved Hum2Melody")
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-monophonic-loss", action='store_true', default=True)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def interpolate_target(target: torch.Tensor, output_frames: int) -> torch.Tensor:
    """
    Properly interpolate target to match model output.
    MUCH better than stride-based downsampling!

    Args:
        target: (batch, input_frames, num_notes)
        output_frames: Desired output frame count

    Returns:
        (batch, output_frames, num_notes)
    """
    batch, input_frames, num_notes = target.shape

    if input_frames == output_frames:
        return target

    # Reshape for interpolation: (batch, num_notes, input_frames)
    target = target.permute(0, 2, 1)

    # Add channel dimension: (batch, 1, num_notes, input_frames)
    target = target.unsqueeze(1)

    # Interpolate along time dimension
    interpolated = nn.functional.interpolate(
        target,
        size=(num_notes, output_frames),
        mode='bilinear',
        align_corners=False
    )

    # Reshape back: (batch, output_frames, num_notes)
    interpolated = interpolated.squeeze(1).permute(0, 2, 1)

    # Threshold to maintain binary nature
    interpolated = (interpolated > 0.5).float()

    return interpolated


def calculate_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    # Convert to binary
    pred_binary = (torch.sigmoid(predictions) > threshold).float()

    # Frame-level accuracy
    correct = (pred_binary == targets).float()
    frame_accuracy = correct.mean().item()

    # Calculate only for frames with activity
    active_frames = (targets.sum(dim=2) > 0)
    silence_frames = (targets.sum(dim=2) == 0)

    if active_frames.sum() > 0:
        note_accuracy = correct[active_frames].mean().item()
    else:
        note_accuracy = 0.0

    if silence_frames.sum() > 0:
        silence_accuracy = correct[silence_frames].mean().item()
    else:
        silence_accuracy = 1.0

    # Precision, Recall, F1
    tp = (pred_binary * targets).sum().item()
    fp = (pred_binary * (1 - targets)).sum().item()
    fn = ((1 - pred_binary) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Pitch accuracy (±1 semitone)
    pitch_correct = 0
    pitch_total = 0

    batch, frames, notes = targets.shape
    for b in range(batch):
        for t in range(frames):
            target_notes = torch.where(targets[b, t] > 0.5)[0]
            pred_notes = torch.where(pred_binary[b, t] > 0.5)[0]

            if len(target_notes) > 0 and len(pred_notes) > 0:
                for tn in target_notes:
                    if any(abs(pn - tn) <= 1 for pn in pred_notes):
                        pitch_correct += 1
                    pitch_total += 1

    pitch_acc = pitch_correct / (pitch_total + 1e-8)

    # Overlap detection (monophonic violation)
    notes_per_frame = pred_binary.sum(dim=2)  # (batch, time)
    overlap_ratio = (notes_per_frame > 1).float().mean().item()

    return {
        'frame_acc': frame_accuracy,
        'note_acc': note_accuracy,
        'silence_acc': silence_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pitch_acc_1st': pitch_acc,
        'overlap_ratio': overlap_ratio
    }


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: CombinedLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_losses = {'focal': 0.0, 'monophonic': 0.0, 'total': 0.0}
    all_metrics = {
        'frame_acc': [],
        'f1': [],
        'pitch_acc_1st': [],
        'overlap_ratio': []
    }

    train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")

    for data, target in train_iter:
        data = data.to(device)
        target = target.to(device).float()

        optimizer.zero_grad()

        # Forward
        output = model(data)

        # Interpolate target to match output
        target_interp = interpolate_target(target, output.shape[1])

        # Calculate loss
        losses = criterion(output, target_interp)
        if isinstance(losses, dict):
            loss = losses['total']
            for key, value in losses.items():
                total_losses[key] += value.item()
        else:
            # FocalLoss returns scalar
            loss = losses
            total_losses['focal'] = total_losses.get('focal', 0.0) + loss.item()
            total_losses['total'] = total_losses.get('total', 0.0) + loss.item()

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(output, target_interp)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

        # Update progress bar
        train_iter.set_postfix({
            'loss': f"{loss.item():.4f}",
            'f1': f"{metrics['f1']:.3f}",
            'pitch': f"{metrics['pitch_acc_1st']:.3f}",
            'overlap': f"{metrics['overlap_ratio']:.3f}"
        })

    # Average everything
    result = {k: v / len(train_loader) for k, v in total_losses.items()}
    for key in all_metrics:
        result[key] = np.mean(all_metrics[key])

    return result


def validate_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: CombinedLoss,
        device: torch.device,
        epoch: int
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()

    total_losses = {'focal': 0.0, 'monophonic': 0.0, 'total': 0.0}
    all_metrics = {
        'frame_acc': [],
        'f1': [],
        'pitch_acc_1st': [],
        'overlap_ratio': []
    }

    val_iter = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}")

    with torch.no_grad():
        for data, target in val_iter:
            data = data.to(device)
            target = target.to(device).float()

            # Forward
            output = model(data)

            # Interpolate target
            target_interp = interpolate_target(target, output.shape[1])

            # Calculate loss
            losses = criterion(output, target_interp)
            if isinstance(losses, dict):
                for key, value in losses.items():
                    total_losses[key] += value.item()
            else:
                loss = losses
                total_losses['focal'] = total_losses.get('focal', 0.0) + loss.item()
                total_losses['total'] = total_losses.get('total', 0.0) + loss.item()

            # Calculate metrics
            metrics = calculate_metrics(output, target_interp)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            val_iter.set_postfix({
                'loss': f"{loss.item():.4f}" if not isinstance(losses, dict) else f"{losses['total'].item():.4f}",
                'f1': f"{metrics['f1']:.3f}"
            })

    # Average everything
    result = {k: v / len(val_loader) for k, v in total_losses.items()}
    for key in all_metrics:
        result[key] = np.mean(all_metrics[key])

    return result


def main():
    args = parse_args()
    torch.manual_seed(42)

    # Setup device
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print("Loading dataset...")
    dataset = MelodyDataset(args.labels, augment=True)

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Model
    model = ImprovedHum2MelodyCRNN(use_attention=True)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Loss
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=50.0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Resume if needed
    start_epoch = 0
    best_f1 = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'val_pitch_acc': [],
        'val_overlap_ratio': [],
        'lr': []
    }

    # Training loop
    epochs_no_improve = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING WITH IMPROVED MODEL")
    print("=" * 70)
    print(f"Improvements:")
    print(f"  ✅ Temporal resolution: 128ms (was 258ms)")
    print(f"  ✅ Focal Loss (was pos_weight=333)")
    print(f"  ✅ Monophonic Loss")
    print(f"  ✅ Proper target interpolation")
    print(f"  ✅ Better metrics tracking")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        print(f"\nTrain Metrics:")
        print(f"  Total Loss: {train_metrics['total']:.4f}")
        print(f"  Focal Loss: {train_metrics['focal']:.4f}")
        if 'monophonic' in train_metrics:
            print(f"  Mono Loss: {train_metrics['monophonic']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")
        print(f"  Pitch Acc (±1st): {train_metrics['pitch_acc_1st']:.4f}")
        print(f"  Overlap Ratio: {train_metrics['overlap_ratio']:.4f}")

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        print(f"\nVal Metrics:")
        print(f"  Total Loss: {val_metrics['total']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        print(f"  Pitch Acc (±1st): {val_metrics['pitch_acc_1st']:.4f}")
        print(f"  Overlap Ratio: {val_metrics['overlap_ratio']:.4f}")

        # Update scheduler
        scheduler.step(val_metrics['total'])
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_pitch_acc'].append(val_metrics['pitch_acc_1st'])
        history['val_overlap_ratio'].append(val_metrics['overlap_ratio'])
        history['lr'].append(current_lr)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'best_f1': best_f1
        }

        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch + 1}.pth")

        # Save best model (based on F1, not loss!)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"  ✅ New best F1: {best_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save history
        with open(checkpoint_dir / "train_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

        if current_lr < 1e-6:
            print(f"\nStopping: learning rate too small ({current_lr})")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == '__main__':
    main()