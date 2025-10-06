
'''
reduce batch_size or num_workers if running out of GPU memory
training history saved under checkpoint

Hum2MelodyCRNN needs to be pytorch nm model
MelodyDataset accepts labels_path,     implement __len__ and __getitem__ returning (input_tensor, target_tensor)
'''


import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import model and dataset modules
from models.hum2melody_model import Hum2MelodyCRNN
from data.melody_dataset import MelodyDataset


# 1. Hyperparameters

BATCH_SIZE = 16  # reduce to 8 if OOM
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.9
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 10
CHECKPOINT_DIR = Path("checkpoints")
HISTORY_FILENAME = CHECKPOINT_DIR / "train_history.json"
SEED = 42
# --------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hum2Melody model")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels/metadata for MelodyDataset")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cpu or cuda:0)")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    return parser.parse_args()

def setup_device(device_override: str = None):
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        prop = torch.cuda.get_device_properties(0)
        print(f"Memory: {prop.total_memory / 1e9:.2f} GB")
    return device

def maybe_pin_memory(device: torch.device):
    return True if device.type == "cuda" else False

def ensure_checkpoint_dir():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(state: Dict[str, Any], filename: Path):
    torch.save(state, str(filename))
    print(f"Saved checkpoint: {filename}")

def save_history(history: Dict[str, list]):
    with open(HISTORY_FILENAME, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {HISTORY_FILENAME}")

def print_gpu_mem(prefix: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"{prefix} GPU mem — allocated: {allocated:.1f} MB, reserved: {reserved:.1f} MB")

def load_checkpoint_if_requested(resume_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print("Warning: failed to load optimizer state:", e)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    return start_epoch, best_val_loss

def main():
    args = parse_args()
    torch.manual_seed(SEED)

    ensure_checkpoint_dir()

    device = setup_device(args.device)
    pin_memory = maybe_pin_memory(device)


    # Dataset loading

    labels_path = args.labels
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    print("Loading dataset...")
    dataset = MelodyDataset(labels_path)
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")

    train_size = int(args.train_split * dataset_size)
    val_size = dataset_size - train_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Invalid train/val split results in empty split. Check TRAIN_SPLIT and dataset size.")
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )


    # Model, loss, optimizer

    model = Hum2MelodyCRNN()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )


    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint_if_requested(args.resume, model, optimizer, device)

    # Training history
    history = {"train_loss": [], "val_loss": [], "lr": []}

    # Early stopping bookkeeping
    epochs_no_improve = 0

    print("Starting training loop...")
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n======== Epoch {epoch+1}/{args.epochs} ========")
            print_gpu_mem(prefix="Start of epoch:")

            # ---- Training ----
            model.train()
            train_loss = 0.0
            train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
            for batch_idx, batch in enumerate(train_iter):
                # Expect dataset __getitem__ to return (input, target)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, target = batch
                else:
                    raise RuntimeError("Dataset must return (data, target) pairs")

                data = data.to(device, non_blocking=pin_memory)
                target = target.to(device, non_blocking=pin_memory).float()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_iter.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = train_loss / len(train_loader)
            print(f"  Train Loss: {train_loss:.6f}")

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            val_iter = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
            with torch.no_grad():
                for batch in val_iter:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        data, target = batch
                    else:
                        raise RuntimeError("Dataset must return (data, target) pairs")

                    data = data.to(device, non_blocking=pin_memory)
                    target = target.to(device, non_blocking=pin_memory).float()

                    output = model(data)
                    v_loss = criterion(output, target).item()
                    val_loss += v_loss
                    val_iter.set_postfix({"val_loss": f"{v_loss:.4f}"})

            val_loss = val_loss / len(val_loader)
            print(f"  Val Loss: {val_loss:.6f}")

            # Scheduler step + record LR
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            # Save periodic checkpoint (epoch)
            ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch+1}.pth"
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, ckpt_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CHECKPOINT_DIR / "best_model.pth"
                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                }, best_path)
                print("  ✓ Saved new best model")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epoch(s)")

            # Save training history
            save_history(history)

            # Print GPU memory usage
            print_gpu_mem(prefix="End of epoch:")

            # Early stopping
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered (no improvement for {args.patience} epochs).")
                break

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving last checkpoint...")
        interrupted_path = CHECKPOINT_DIR / f"interrupted_epoch_{epoch+1}.pth"
        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
        }, interrupted_path)
    except Exception as e:
        print("Unhandled exception during training:", e)
        raise

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    save_history(history)

if __name__ == "__main__":
    main()
