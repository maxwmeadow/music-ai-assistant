#!/usr/bin/env python3
"""
Training script for DrumClassifierCNN using CNN-detected onsets.

This trains the classifier on data preprocessed with the CNN onset detector
to ensure train/test consistency.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from models.drum_classifier import create_model
from data.drum_dataset_cnn import create_dataloaders_cnn, DrumDatasetCNN


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (specs, labels) in enumerate(pbar):
        specs, labels = specs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class statistics
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, (specs, labels) in enumerate(pbar):
            specs, labels = specs.to(device), labels.to(device)

            # Forward pass
            outputs = model(specs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    # Per-class accuracy
    class_acc = {}
    for i, class_name in enumerate(DrumDatasetCNN.CLASS_NAMES):
        if class_total[i] > 0:
            class_acc[class_name] = 100. * class_correct[i] / class_total[i]
        else:
            class_acc[class_name] = 0.0

    return epoch_loss, epoch_acc, class_acc


def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model_cnn.pth'
        torch.save(checkpoint, best_path)
        print(f'✓ Saved best model (val_acc: {val_acc:.2f}%)')

    return checkpoint_path


def train(args):
    """Main training function."""
    print("="*70)
    print("Training DrumClassifierCNN (CNN-based preprocessing)")
    print("="*70)
    print()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders_cnn(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampling=args.weighted_sampling,
        pin_memory=(device.type == 'cuda')
    )
    print()

    # Create model
    print("Creating model...")
    model = create_model(num_classes=3, dropout=args.dropout, device=device)
    print(f"Total parameters: {model.get_num_params():,}")
    print()

    # Test model with actual data shape
    sample_batch, _ = next(iter(train_loader))
    print(f"Input data shape: {sample_batch.shape}")
    test_output = model(sample_batch[:2].to(device))
    print(f"Model output shape: {test_output.shape}")
    print()

    # Loss and optimizer
    if args.use_class_weights:
        class_weights = train_loader.dataset.get_class_weights().to(device)
        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Tensorboard
    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir)
    print(f"Tensorboard logs: {log_dir}")
    print()

    # Training loop
    print("="*70)
    print("Starting training")
    print("="*70)
    print()

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Logging
        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Per-class Val Acc:')
        for class_name, acc in class_acc.items():
            print(f'    {class_name}: {acc:.2f}%')
        print()

        # Tensorboard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        for class_name, acc in class_acc.items():
            writer.add_scalar(f'Val Acc/{class_name}', acc, epoch)

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % args.save_every == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_acc, args.checkpoint_dir, is_best)

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            print(f'Best val accuracy: {best_val_acc:.2f}%')
            break

    print()
    print("="*70)
    print("Training complete!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print()

    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'config': vars(args),
        'note': 'Trained on CNN-detected onsets for train/test consistency'
    }

    results_path = Path(args.checkpoint_dir) / 'results_cnn.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DrumClassifierCNN on CNN-detected data')

    # Data
    parser.add_argument('--data-dir', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/classifier_data_cnn',
                       help='Directory with CNN-preprocessed .npz files')

    # Model
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Training
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--weighted-sampling', action='store_true',
                       help='Use weighted sampling for training')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights in loss function')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Early stopping patience')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/classifier_checkpoints_cnn',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log-dir', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/classifier_logs_cnn',
                       help='Tensorboard log directory')

    args = parser.parse_args()

    train(args)
