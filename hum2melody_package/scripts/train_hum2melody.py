"""
Train Hum2Melody - Complete Training Script

This script consolidates all training features:
1. Standard training mode (single dataset)
2. Curriculum learning mode (synthetic -> real)
3. Support for both basic and enhanced models
4. Automatic pos_weight computation
5. Per-batch scheduler stepping
6. Proper metric calculation
7. Comprehensive logging

Usage:
  # Basic training
  python train_hum2melody.py --labels data/manifest.json --batch-size 16 --epochs 50
  
  # Enhanced model with all features
  python train_hum2melody.py --labels data/manifest.json --use-enhanced --use-onset-features --use-musical-context
  
  # Curriculum learning
  python train_hum2melody.py --labels data/real.json --curriculum --synthetic-clean data/clean.json --synthetic-noisy data/noisy.json
"""

import sys
from pathlib import Path
import os
import json
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Import our consolidated modules
from models.hum2melody_model import (
    ImprovedHum2MelodyWithOnsets,
    EnhancedHum2MelodyModel,
    DualPathHum2MelodyModel,
    MultiTaskLoss,
    EnhancedMultiTaskLoss
)
from data.melody_dataset import MelodyDataset, EnhancedMelodyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hum2Melody Model")
    
    # Data
    parser.add_argument("--labels", type=str, required=True, help="Path to labels manifest")
    parser.add_argument("--train-split", type=float, default=0.9)
    
    # Model selection
    parser.add_argument("--use-enhanced", action='store_true',
                        help="Use enhanced model (4 heads) instead of basic (2 heads)")
    parser.add_argument("--use-dual-path", action='store_true',
                        help="Use dual-path architecture (separate harmonic and temporal LSTMs)")

    # Enhanced model features
    parser.add_argument("--use-pretrained", action='store_true',
                        help="Use pretrained wav2vec2 features")
    parser.add_argument("--use-onset-features", action='store_true',
                        help="Use onset-strength features")
    parser.add_argument("--use-musical-context", action='store_true',
                        help="Use musical context features")
    parser.add_argument("--use-multi-scale", action='store_true',
                        help="Use multi-scale temporal encoder")
    parser.add_argument("--use-transition-model", action='store_true',
                        help="Use musical transition model")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use-amp", action='store_true', help="Use mixed precision")
    
    # Model architecture
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-bins", type=int, default=84, help="Number of CQT bins")
    
    # Loss weights (BALANCED for preventing onset head death)
    parser.add_argument("--frame-weight", type=float, default=1.0,
                        help="Frame loss weight (reduced from 5.0 for balance)")
    parser.add_argument("--onset-weight", type=float, default=1.0,
                        help="Onset loss weight (reduced to 1.0 with dual-LSTM - internal weight is 5.0, effective 5x)")
    parser.add_argument("--offset-weight", type=float, default=0.5)
    parser.add_argument("--f0-weight", type=float, default=1.0)
    parser.add_argument("--mono-weight", type=float, default=0.02)
    parser.add_argument("--sparsity-weight", type=float, default=0.0,
                        help="DISABLED temporarily (0.0) - was pushing collapsed predictions further down")
    parser.add_argument("--temporal-loss-scale", type=float, default=0.1,
                        help="Scale factor for temporal path (onset/offset) loss to dampen gradients (default: 0.1)")

    # Warmup - NO LONGER NEEDED with dual-LSTM architecture
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Warmup disabled by default (0) - dual-LSTM architecture eliminates task interference")
    parser.add_argument("--no-warmup", action='store_true',
                        help="Disable warmup (use frame_weight from start)")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-weights-only", action='store_true',
                        help="When resuming, load model weights only (NOT optimizer state) - for transfer learning")

    # Stage 2 training options
    parser.add_argument("--freeze-backbone", action='store_true',
                        help="Freeze shared LSTM, frame head, and f0 head (for Stage 2 heads-only training)")
    parser.add_argument("--freeze-harmonic-path", action='store_true',
                        help="Freeze harmonic path (CNN + harmonic LSTM + frame/f0 heads) for dual-path training")
    parser.add_argument("--lr-onset", type=float, default=None,
                        help="Learning rate for onset/offset heads (overrides --lr * 0.5)")
    parser.add_argument("--oversample-onsets", action='store_true',
                        help="Oversample batches containing onsets (helps with class imbalance)")
    parser.add_argument("--auto-rollback", action='store_true',
                        help="Automatically rollback if Frame F1 drops significantly")
    parser.add_argument("--rollback-threshold", type=float, default=0.10,
                        help="Frame F1 drop threshold for rollback (default: 0.10 = 10%%)")
    parser.add_argument("--rollback-grace-epochs", type=int, default=3,
                        help="Number of epochs before checking rollback (default: 3)")
    parser.add_argument("--rollback-terminate", action='store_true',
                        help="Terminate training after rollback (default: continue with restored weights)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument("--onset-warmup-schedule", nargs=3, type=float,
                        default=[0.25, 0.5, 1.0],
                        help="Onset weight warmup schedule [start, mid, final] over epochs")
    parser.add_argument("--onset-warmup-epochs", type=int, default=15,
                        help="Total epochs for onset weight warmup")
    
    # Curriculum learning
    parser.add_argument("--curriculum", action='store_true',
                        help="Use curriculum learning")
    parser.add_argument("--synthetic-clean", type=str, default=None)
    parser.add_argument("--synthetic-noisy", type=str, default=None)
    parser.add_argument("--curriculum-epochs", nargs=3, type=int,
                        default=[10, 10, 30],
                        help="Epochs for [clean, noisy, real]")
    
    # Pos-weight computation
    parser.add_argument("--compute-pos-weights", action='store_true',
                        help="Compute pos_weights from dataset")
    parser.add_argument("--pos-weight-batches", type=int, default=100)
    
    return parser.parse_args()


def compute_pos_weights(
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
    use_enhanced: bool = False
) -> Dict[str, float]:
    """
    Compute pos_weight for onset (and offset if enhanced) from dataset statistics.
    """
    print(f"\n{'='*60}")
    print("COMPUTING POS_WEIGHTS FROM DATASET")
    print(f"{'='*60}")
    print(f"Analyzing {max_batches} batches...")
    
    total_frames = 0
    total_onset_pos = 0
    total_offset_pos = 0 if use_enhanced else None
    
    for i, (data, targets) in enumerate(tqdm(dataloader, desc="Analyzing", total=max_batches)):
        if i >= max_batches:
            break
        
        batch_size, time_steps = targets['onset'].shape
        total_frames += batch_size * time_steps
        
        total_onset_pos += targets['onset'].sum().item()
        
        if use_enhanced and 'offset' in targets:
            total_offset_pos += targets['offset'].sum().item()
    
    # Calculate pos_weight
    onset_neg = total_frames - total_onset_pos
    onset_pw = onset_neg / (total_onset_pos + 1e-9)
    onset_pw = max(1.0, min(onset_pw, 10.0))
    
    result = {'onset': onset_pw}
    
    if use_enhanced and total_offset_pos is not None:
        offset_neg = total_frames - total_offset_pos
        offset_pw = offset_neg / (total_offset_pos + 1e-9)
        offset_pw = max(1.0, min(offset_pw, 8.0))
        result['offset'] = offset_pw
    
    print(f"\nDataset statistics:")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Onset positive: {total_onset_pos:,} ({100*total_onset_pos/total_frames:.2f}%)")
    print(f"  Computed onset pos_weight: {result['onset']:.2f}")
    
    if 'offset' in result:
        print(f"  Offset positive: {total_offset_pos:,} ({100*total_offset_pos/total_frames:.2f}%)")
        print(f"  Computed offset pos_weight: {result['offset']:.2f}")
    
    print(f"{'='*60}\n")
    
    return result


def interpolate_target(
    target: torch.Tensor,
    target_name: str,
    src_frames: int,
    dst_frames: int,
    device: torch.device
) -> torch.Tensor:
    """
    Resample target from src_frames -> dst_frames.

    CRITICAL: Different target types need different interpolation modes!

    - Onset/Offset (SPARSE binary): Use 'nearest' to preserve rare events
      - Events are 5-10% density
      - Bilinear averaging dilutes them below threshold
      - Example: [0,1,0,0,0] → nearest → [0,1,0] ✓
      - Example: [0,1,0,0,0] → bilinear → [0.2,0.4,0.1] → threshold → [0,0,0] ✗

    - Frame (DENSE multi-hot): Use 'bilinear' for smooth boundaries
      - Notes sustained over many frames (high density)
      - Bilinear provides smooth transitions at note boundaries
      - Nearest can cause misalignment artifacts
      - Example: [1,1,1,1,0,0] → bilinear → [1.0,0.75] → threshold → [1,1] ✓
      - Example: [1,1,1,1,0,0] → nearest → [1,0] (boundary error) ✗

    - F0 (continuous): Use 'bilinear' for smooth values
    """
    t = target.to(device).float()

    # Normalize shape to (batch, channels, time) for interpolate
    if t.dim() == 2:
        t = t.unsqueeze(-1)

    batch, s_frames, channels = t.shape
    t = t.permute(0, 2, 1)  # (batch, channels, time)
    t = t.unsqueeze(-1)  # (batch, channels, time, 1)

    # Choose interpolation mode based on target type
    if target_name in ['onset', 'offset']:
        # SPARSE binary: use nearest to preserve rare events
        interpolation_mode = 'nearest'
        is_binary = True
    elif target_name == 'frame':
        # DENSE multi-hot: use bilinear for smooth boundaries
        interpolation_mode = 'bilinear'
        is_binary = True
    else:  # f0
        # Continuous: use bilinear
        interpolation_mode = 'bilinear'
        is_binary = False

    interpolated = F.interpolate(
        t, size=(dst_frames, 1), mode=interpolation_mode,
        align_corners=False if interpolation_mode == 'bilinear' else None
    )
    interpolated = interpolated.squeeze(-1)  # (batch, channels, dst_frames)
    interpolated = interpolated.permute(0, 2, 1)  # (batch, dst_frames, channels)

    if is_binary:
        # Apply threshold to ensure binary values
        interpolated = (interpolated > 0.5).float()
        if channels == 1:
            interpolated = interpolated.squeeze(-1)

    return interpolated


def calculate_metrics(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Calculate all metrics from predictions and targets.
    
    Returns dict with:
        - frame_f1: Frame-level F1 score
        - onset_f1: Onset F1 score
        - offset_f1: Offset F1 score (if available)
        - f0_mae: F0 mean absolute error (if available)
        - pitch_acc: Pitch accuracy (if available)
    """
    out = {}
    
    # Helper to get prediction tensor
    def _get_pred_tensor(key_logits: str, key_prob: str):
        if key_logits in preds and preds[key_logits] is not None:
            return preds[key_logits].detach()
        if key_prob in preds and preds[key_prob] is not None:
            return preds[key_prob].detach()
        return None
    
    # Frame metrics
    frame_pred = _get_pred_tensor('frame_logits', 'frame')
    frame_targ = targets.get('frame', None)
    
    if frame_pred is None or frame_targ is None:
        out['frame_f1'] = 0.0
    else:
        fp = frame_pred.detach().cpu()
        ft = frame_targ.detach().cpu()
        
        if fp.dim() == 2:
            fp = fp.unsqueeze(-1)
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1)
        
        # Detect if logits or probs
        is_logits = (fp.max() > 2.0) or (fp.min() < -2.0)
        if is_logits:
            prob = torch.sigmoid(fp)
        else:
            prob = fp.clamp(0.0, 1.0)
        
        # Frame detection uses standard 0.5 threshold (not sparse)
        pred_bin = (prob > 0.5).float()

        # Micro F1
        p_flat = pred_bin.numpy().reshape(-1)
        t_flat = (ft.numpy().reshape(-1) > 0.5).astype(float)
        
        tp = float(((p_flat == 1) & (t_flat == 1)).sum())
        fp_cnt = float(((p_flat == 1) & (t_flat == 0)).sum())
        fn = float(((p_flat == 0) & (t_flat == 1)).sum())
        prec = tp / (tp + fp_cnt + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        out['frame_f1'] = float(f1)
    
    # Binary metric helper
    def _binary_metric(pred_keys, target_key):
        pred_tensor = _get_pred_tensor(pred_keys[0], pred_keys[1])
        targ = targets.get(target_key, None)
        if pred_tensor is None or targ is None:
            return 0.0
        
        pt = pred_tensor.detach().cpu()
        tt = targ.detach().cpu()
        
        if pt.dim() == 3 and pt.shape[-1] == 1:
            pt = pt.squeeze(-1)
        if tt.dim() == 3 and tt.shape[-1] == 1:
            tt = tt.squeeze(-1)
        
        is_logits_local = (pt.max() > 2.0) or (pt.min() < -2.0)
        if is_logits_local:
            prob_local = torch.sigmoid(pt)
        else:
            prob_local = pt.clamp(0.0, 1.0)
        
        # EXTREMELY low threshold for onset/offset to catch any predictions at all
        # Diagnostic showed predictions in 0.02-0.05 range even with corrected pos_weight
        # Starting with 0.05 to see if model is actually learning
        threshold = 0.05 if target_key in ['onset', 'offset'] else 0.5
        pred_bin_local = (prob_local > threshold).float()

        p_flat = pred_bin_local.numpy().reshape(-1)
        t_flat = (tt.numpy().reshape(-1) > 0.5).astype(float)
        tp = float(((p_flat == 1) & (t_flat == 1)).sum())
        fp_cnt = float(((p_flat == 1) & (t_flat == 0)).sum())
        fn = float(((p_flat == 0) & (t_flat == 1)).sum())
        prec = tp / (tp + fp_cnt + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        return float(f1)

    out['onset_f1'] = _binary_metric(('onset_logits', 'onset'), 'onset')
    out['offset_f1'] = _binary_metric(('offset_logits', 'offset'), 'offset')
    
    # F0 MAE
    if 'f0' in preds and 'f0' in targets:
        try:
            pred_f0 = preds['f0'].detach().cpu()
            targ_f0 = targets['f0'].detach().cpu()
            if pred_f0.ndim == targ_f0.ndim and pred_f0.shape[-1] >= 2 and targ_f0.shape[-1] >= 2:
                voiced_mask = (targ_f0[..., 1] > 0.5).float()
                if voiced_mask.sum() > 0:
                    mae = torch.abs((pred_f0[..., 0] - targ_f0[..., 0]) * voiced_mask).sum() / (voiced_mask.sum() + 1e-9)
                    out['f0_mae'] = float(mae.item())
                else:
                    out['f0_mae'] = 0.0
            else:
                out['f0_mae'] = 0.0
        except Exception:
            out['f0_mae'] = 0.0
    else:
        out['f0_mae'] = 0.0
    
    # Pitch accuracy
    try:
        if ('frame_logits' in preds) or ('frame' in preds):
            frame_pred2 = _get_pred_tensor('frame_logits', 'frame')
            if frame_pred2 is not None:
                fp2 = frame_pred2.detach().cpu()
                if (fp2.max() > 2.0) or (fp2.min() < -2.0):
                    prob2 = torch.sigmoid(fp2)
                else:
                    prob2 = fp2.clamp(0.0, 1.0)
                pred_idx = torch.argmax(prob2, dim=-1).numpy()
                targ_idx = torch.argmax(targets['frame'].detach().cpu(), dim=-1).numpy()
                correct = (pred_idx == targ_idx).astype(float)
                acc = float(correct.sum() / correct.size)
                out['pitch_acc'] = acc
            else:
                out['pitch_acc'] = 0.0
        else:
            out['pitch_acc'] = 0.0
    except Exception:
        out['pitch_acc'] = 0.0
    
    # Ensure all keys present
    for k in ['frame_f1', 'onset_f1', 'offset_f1', 'f0_mae', 'pitch_acc']:
        if k not in out:
            out[k] = 0.0
    
    return out


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    log_file: Optional[Path] = None,
    grad_clip: float = 1.0,
    monitor_gradients: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    print(f"\n{'='*60}")
    print(f"TRAINING EPOCH {epoch + 1}")
    print(f"{'='*60}")
    
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_losses = {'frame': 0.0, 'onset': 0.0, 'offset': 0.0, 'f0': 0.0, 'mono': 0.0, 'total': 0.0}
    all_metrics = {
        'frame_f1': [],
        'onset_f1': [],
        'offset_f1': [],
        'f0_mae': [],
        'pitch_acc': []
    }

    metrics = {'frame_f1': 0.0, 'onset_f1': 0.0}

    # Onset death detector - warn if onset head is dying
    consecutive_zero_onset_f1 = 0
    onset_death_threshold = 20  # Warn after 20 consecutive batches with 0 onset F1
    
    train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")
    
    for batch_idx, (data, targets) in enumerate(train_iter):
        data = data.to(device)
        targets = {k: v.to(device).float() for k, v in targets.items()}
        
        optimizer.zero_grad()
        
        # Forward
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(data)
            
            # Resample targets to match model output
            output_frames = outputs['frame'].shape[1]
            target_frames = targets['frame'].shape[1]
            
            if output_frames != target_frames:
                targets['frame'] = interpolate_target(
                    targets['frame'], 'frame', target_frames, output_frames, device
                )
                targets['onset'] = interpolate_target(
                    targets['onset'], 'onset', target_frames, output_frames, device
                )
                if 'offset' in targets:
                    targets['offset'] = interpolate_target(
                        targets['offset'], 'offset', target_frames, output_frames, device
                    )
                if 'f0' in targets:
                    targets['f0'] = interpolate_target(
                        targets['f0'], 'f0', target_frames, output_frames, device
                    )
            
            losses = criterion(outputs, targets)
            loss = losses['total']
        
        # Backward with gradient clipping
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Monitor per-component gradient norms (before clipping)
            # Shared architecture: backbone is shared, only heads differ
            if batch_idx < 10:
                shared_grad_norm = 0.0
                frame_head_grad_norm = 0.0
                onset_head_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        if 'shared' in name or 'cnn' in name:
                            shared_grad_norm += param_norm ** 2
                        elif 'frame_head' in name or 'f0_head' in name:
                            frame_head_grad_norm += param_norm ** 2
                        elif 'onset_head' in name or 'offset_head' in name:
                            onset_head_grad_norm += param_norm ** 2
                shared_grad_norm = shared_grad_norm ** 0.5
                frame_head_grad_norm = frame_head_grad_norm ** 0.5
                onset_head_grad_norm = onset_head_grad_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Monitor per-component gradient norms (before clipping)
            # Shared architecture: backbone is shared, only heads differ
            if batch_idx < 10 or monitor_gradients:
                shared_grad_norm = 0.0
                frame_head_grad_norm = 0.0
                onset_head_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        if 'shared' in name or 'cnn' in name:
                            shared_grad_norm += param_norm ** 2
                        elif 'frame_head' in name or 'f0_head' in name:
                            frame_head_grad_norm += param_norm ** 2
                        elif 'onset_head' in name or 'offset_head' in name:
                            onset_head_grad_norm += param_norm ** 2
                shared_grad_norm = shared_grad_norm ** 0.5
                frame_head_grad_norm = frame_head_grad_norm ** 0.5
                onset_head_grad_norm = onset_head_grad_norm ** 0.5

                # Check for gradient explosion (temporal vs harmonic)
                if monitor_gradients and shared_grad_norm > 0:
                    temporal_harmonic_ratio = onset_head_grad_norm / (frame_head_grad_norm + 1e-9)
                    if temporal_harmonic_ratio > 200:
                        print(f"\n⚠️ GRADIENT EXPLOSION DETECTED!")
                        print(f"   Temporal/Harmonic ratio: {temporal_harmonic_ratio:.1f} (>200×)")
                        print(f"   Consider reducing temporal_loss_scale or onset_weight")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        
        # Step scheduler per batch
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate losses
        for k in ['frame', 'onset', 'offset', 'f0', 'mono', 'total']:
            if k in losses:
                val = losses[k].item() if isinstance(losses[k], torch.Tensor) else float(losses[k])
                total_losses[k] += val
        
        # DETAILED ONSET DEBUGGING (first 10 batches) - BEFORE metrics calculation
        # USE FLUSH=TRUE TO FORCE OUTPUT IN SLURM
        if batch_idx < 10:
            print(f"\n{'='*60}", flush=True)
            print(f"[BATCH {batch_idx}] ONSET DEBUGGING", flush=True)
            print(f"{'='*60}", flush=True)

            try:
                onset_logits = outputs.get('onset_logits', outputs.get('onset'))
                onset_targets = targets['onset']

                if onset_logits is not None:
                    import numpy as np
                    onset_probs = torch.sigmoid(onset_logits).detach().cpu().numpy()
                    onset_targs = onset_targets.detach().cpu().numpy()

                    # Flatten
                    probs_flat = onset_probs.flatten()
                    targs_flat = onset_targs.flatten()

                    # Stats
                    true_onset_mask = targs_flat > 0.5
                    num_true_onsets = true_onset_mask.sum()

                    print(f"True onsets in batch: {num_true_onsets}/{len(targs_flat)} ({100*num_true_onsets/len(targs_flat):.2f}%)", flush=True)
                    print(f"Pred probs - min:{probs_flat.min():.6f} max:{probs_flat.max():.6f} mean:{probs_flat.mean():.6f}", flush=True)
                    print(f"Pred logits - min:{onset_logits.min().item():.4f} max:{onset_logits.max().item():.4f}", flush=True)

                    if num_true_onsets > 0:
                        true_onset_probs = probs_flat[true_onset_mask]
                        print(f"AT TRUE ONSETS - min:{true_onset_probs.min():.6f} max:{true_onset_probs.max():.6f} mean:{true_onset_probs.mean():.6f} std:{true_onset_probs.std():.6f}", flush=True)

                        # Distribution percentiles at true onsets
                        percentiles = np.percentile(true_onset_probs, [1, 5, 25, 50, 75, 95, 99])
                        print(f"  Percentiles [1,5,25,50,75,95,99]: {[f'{p:.3f}' for p in percentiles]}", flush=True)

                        # How many would pass threshold?
                        for thresh in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
                            num_detected = (true_onset_probs > thresh).sum()
                            print(f"  @{thresh}: {num_detected}/{num_true_onsets} = {100*num_detected/num_true_onsets:.1f}%", flush=True)

                    print(f"Onset loss: {losses['onset'].item():.6f}", flush=True)

                    # Phase 2: Monitor sparsity regularization
                    if 'sparsity' in losses:
                        print(f"Sparsity loss: {losses['sparsity'].item():.6f}", flush=True)
                    if 'mean_onset_pred' in losses:
                        mean_pred = losses['mean_onset_pred'].item()
                        print(f"Mean onset pred: {mean_pred:.4f} (target: 0.0336)", flush=True)

                    # Gradient norm monitoring (shared-LSTM architecture)
                    if 'shared_grad_norm' in locals() and 'frame_head_grad_norm' in locals():
                        print(f"\nGradient Norms (pre-clipping):", flush=True)
                        print(f"  Shared backbone: {shared_grad_norm:.4f}", flush=True)
                        print(f"  Frame/F0 heads: {frame_head_grad_norm:.4f}", flush=True)
                        print(f"  Onset/Offset heads: {onset_head_grad_norm:.4f}", flush=True)
                        # Check if onset gradients dominate
                        if onset_head_grad_norm > shared_grad_norm * 3.0:
                            print(f"  ⚠️ Onset head gradients dominating! Consider lower temporal_loss_scale.", flush=True)
                        elif onset_head_grad_norm < shared_grad_norm * 0.1:
                            print(f"  ⚠️ Onset head gradients too weak! Consider higher temporal_loss_scale.", flush=True)
                else:
                    print("⚠️ onset_logits not found in outputs!", flush=True)
                    print(f"Output keys: {list(outputs.keys())}", flush=True)
            except Exception as e:
                print(f"⚠️ Debug error: {e}", flush=True)
                import traceback
                traceback.print_exc()

            print(f"{'='*60}\n", flush=True)

        # Calculate metrics periodically
        if batch_idx % 5 == 0:
            batch_metrics = calculate_metrics(outputs, targets, device=device)
            all_metrics['frame_f1'].append(batch_metrics['frame_f1'])
            all_metrics['onset_f1'].append(batch_metrics['onset_f1'])
            all_metrics['offset_f1'].append(batch_metrics['offset_f1'])
            all_metrics['f0_mae'].append(batch_metrics['f0_mae'])
            all_metrics['pitch_acc'].append(batch_metrics['pitch_acc'])

            metrics = {
                'frame_f1': batch_metrics['frame_f1'],
                'onset_f1': batch_metrics['onset_f1'],
            }

            # ONSET DEATH DETECTOR
            if batch_metrics['onset_f1'] == 0.0:
                consecutive_zero_onset_f1 += 1
                if consecutive_zero_onset_f1 == onset_death_threshold:
                    print(f"\n⚠️  WARNING: Onset F1 has been 0.0 for {onset_death_threshold} consecutive batches!")
                    print(f"⚠️  Onset head may be dying. Current onset loss: {losses['onset'].item():.6f}")
                    print(f"⚠️  Frame F1: {batch_metrics['frame_f1']:.4f}")
            else:
                consecutive_zero_onset_f1 = 0  # Reset counter if we see non-zero

        # Update progress bar
        train_iter.set_postfix({
            'loss': f"{loss.item():.4f}",
            'frame_f1': f"{metrics['frame_f1']:.3f}",
            'onset_f1': f"{metrics['onset_f1']:.3f}"
        })
        
        # Log per-batch losses
        if log_file is not None and batch_idx % 10 == 0:
            with open(log_file, "a") as f:
                f.write(json.dumps({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss_total': float(total_losses['total'] / (batch_idx + 1)),
                    'loss_frame': float(total_losses['frame'] / (batch_idx + 1)),
                    'loss_onset': float(total_losses['onset'] / (batch_idx + 1)),
                }) + "\n")
    
    # Finalize epoch metrics
    num_batches = len(train_loader)
    avg_losses = {k: (total_losses[k] / max(1, num_batches)) for k in total_losses}
    
    epoch_metrics = {
        'total': avg_losses['total'],
        'frame': avg_losses['frame'],
        'onset': avg_losses['onset'],
        'offset': avg_losses['offset'],
        'f0': avg_losses['f0'],
        'mono': avg_losses['mono'],
        'frame_f1': float(np.mean(all_metrics['frame_f1'])) if all_metrics['frame_f1'] else 0.0,
        'onset_f1': float(np.mean(all_metrics['onset_f1'])) if all_metrics['onset_f1'] else 0.0,
        'offset_f1': float(np.mean(all_metrics['offset_f1'])) if all_metrics['offset_f1'] else 0.0,
        'f0_mae': float(np.mean(all_metrics['f0_mae'])) if all_metrics['f0_mae'] else 0.0,
        'pitch_acc': float(np.mean(all_metrics['pitch_acc'])) if all_metrics['pitch_acc'] else 0.0,
    }
    
    print(f"\nEpoch {epoch + 1} Training Complete:")
    print(f"  Loss: {epoch_metrics['total']:.4f}")
    print(f"  Frame F1: {epoch_metrics['frame_f1']:.4f}")
    print(f"  Onset F1: {epoch_metrics['onset_f1']:.4f}")
    if epoch_metrics['offset_f1'] > 0:
        print(f"  Offset F1: {epoch_metrics['offset_f1']:.4f}")
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch."""
    print(f"\n{'='*60}")
    print(f"VALIDATING EPOCH {epoch + 1}")
    print(f"{'='*60}")
    
    model.eval()
    
    total_losses = {'frame': 0.0, 'onset': 0.0, 'offset': 0.0, 'f0': 0.0, 'mono': 0.0, 'total': 0.0}
    all_metrics = {
        'frame_f1': [],
        'onset_f1': [],
        'offset_f1': [],
        'f0_mae': [],
        'pitch_acc': []
    }
    
    val_iter = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}")
    
    with torch.no_grad():
        for data, targets in val_iter:
            data = data.to(device)
            targets = {k: v.to(device).float() for k, v in targets.items()}
            
            predictions = model(data)
            
            # Resample targets
            output_frames = predictions['frame'].shape[1]
            target_frames = targets['frame'].shape[1]
            
            if output_frames != target_frames:
                targets['frame'] = interpolate_target(
                    targets['frame'], 'frame', target_frames, output_frames, device
                )
                targets['onset'] = interpolate_target(
                    targets['onset'], 'onset', target_frames, output_frames, device
                )
                if 'offset' in targets:
                    targets['offset'] = interpolate_target(
                        targets['offset'], 'offset', target_frames, output_frames, device
                    )
                if 'f0' in targets:
                    targets['f0'] = interpolate_target(
                        targets['f0'], 'f0', target_frames, output_frames, device
                    )
            
            losses = criterion(predictions, targets)
            
            for key, value in losses.items():
                if key in total_losses:
                    total_losses[key] += value.item() if isinstance(value, torch.Tensor) else float(value)
            
            metrics = calculate_metrics(predictions, targets, device)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Finalize
    result = {k: v / len(val_loader) for k, v in total_losses.items()}
    for key in all_metrics:
        result[key] = np.mean(all_metrics[key])
    
    print(f"\nEpoch {epoch + 1} Validation Complete:")
    print(f"  Loss: {result['total']:.4f}")
    print(f"  Frame F1: {result['frame_f1']:.4f}")
    print(f"  Onset F1: {result['onset_f1']:.4f}")
    if result['offset_f1'] > 0:
        print(f"  Offset F1: {result['offset_f1']:.4f}")
    
    return result


def main():
    args = parse_args()
    torch.manual_seed(42)
    
    # Device setup
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\n{'='*70}")
    print("HUM2MELODY TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {'Enhanced (4 heads)' if args.use_enhanced else 'Basic (2 heads)'}")
    
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
    
    if args.use_enhanced:
        print(f"\nEnhanced Features:")
        if args.use_pretrained:
            print(f"  ✅ Pretrained embeddings")
        if args.use_onset_features:
            print(f"  ✅ Onset features")
        if args.use_musical_context:
            print(f"  ✅ Musical context")
        if args.use_multi_scale:
            print(f"  ✅ Multi-scale encoder")
        if args.use_transition_model:
            print(f"  ✅ Transition model")
    
    print(f"{'='*70}\n")
    
    # Create dataset
    print("Creating dataset...")
    if args.use_enhanced:
        dataset = EnhancedMelodyDataset(
            labels_path=args.labels,
            n_bins=args.n_bins,
            augment=True,
            spec_augment=True,
            use_pretrained=args.use_pretrained,
            use_onset_features=args.use_onset_features,
            use_musical_context=args.use_musical_context
        )
    else:
        dataset = MelodyDataset(
            labels_path=args.labels,
            n_bins=args.n_bins,
            augment=True,
            spec_augment=True
        )
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Detect input dimensions
    sample_input, _ = dataset[0]
    input_channels = sample_input.shape[-1]
    print(f"Input channels: {input_channels}")
    
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
    
    # Compute pos_weights if requested
    pos_weights = None
    if args.compute_pos_weights:
        pos_weights = compute_pos_weights(
            train_loader,
            device,
            max_batches=args.pos_weight_batches,
            use_enhanced=args.use_enhanced
        )
    
    # Create model
    print("\nCreating model...")
    if args.use_dual_path:
        model = DualPathHum2MelodyModel(
            n_bins=args.n_bins,
            input_channels=input_channels,
            hidden_size=args.hidden_size,
            use_attention=False  # Simplified for dual-path
        )
    elif args.use_enhanced:
        model = EnhancedHum2MelodyModel(
            n_bins=args.n_bins,
            input_channels=input_channels,
            hidden_size=args.hidden_size,
            use_attention=True,
            use_multi_scale=args.use_multi_scale,
            use_transition_model=args.use_transition_model
        )
    else:
        model = ImprovedHum2MelodyWithOnsets(
            n_bins=args.n_bins,
            hidden_size=args.hidden_size
        )

    model = model.to(device)
    print(f"✅ Model created: {model.count_parameters():,} parameters")
    
    # Create loss criterion
    print("\nCreating loss criterion...")
    if args.use_dual_path or args.use_enhanced:
        criterion = EnhancedMultiTaskLoss(
            frame_weight=args.frame_weight,
            onset_weight=args.onset_weight,
            offset_weight=args.offset_weight,
            f0_weight=args.f0_weight,
            mono_weight=args.mono_weight,
            sparsity_weight=args.sparsity_weight,  # Phase 2: prevent predict-everywhere-high
            temporal_loss_scale=args.temporal_loss_scale,  # Direct gradient dampening for temporal path
            use_improved_onset_loss=True,
            device=str(device)
        ).to(device)
    else:
        criterion = MultiTaskLoss(
            frame_weight=args.frame_weight,
            onset_weight=args.onset_weight,
            mono_weight=args.mono_weight,
            device=str(device)
        ).to(device)
    
    # Update pos_weights if computed
    if pos_weights is not None:
        print(f"Updating pos_weights in criterion...")
        if hasattr(criterion, 'onset_offset_criterion'):
            criterion.onset_offset_criterion.onset_criterion.pos_weight = torch.tensor(
                pos_weights['onset'], device=device
            )
            if 'offset' in pos_weights:
                criterion.onset_offset_criterion.offset_criterion.pos_weight = torch.tensor(
                    pos_weights['offset'], device=device
                )
        else:
            criterion.onset_criterion.pos_weight = torch.tensor(
                pos_weights['onset'], device=device
            )
    
    print("✅ Loss criterion created")

    # Stage 2: Freeze backbone if requested
    if args.freeze_backbone:
        print(f"\n🔒 FREEZING BACKBONE (Stage 2 heads-only training)")
        frozen_count = 0
        for name, param in model.named_parameters():
            # Freeze shared components and frame/f0 heads
            if 'onset_head' not in name and 'offset_head' not in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"   Frozen {frozen_count} parameter groups")
        print(f"   Training ONLY onset/offset heads")

    # Dual-path: Freeze harmonic path if requested
    if args.freeze_harmonic_path:
        if hasattr(model, 'freeze_harmonic_path'):
            frozen_params = model.freeze_harmonic_path()
        else:
            raise ValueError("--freeze-harmonic-path requires --use-dual-path model")

    # Create optimizer with separate LR for onset/offset heads (stability)
    # For dual-path: temporal path gets onset_lr, harmonic path gets lr
    # For shared-LSTM: onset/offset heads get onset_lr, rest gets lr
    shared_backbone_params = []
    frame_f0_params = []
    onset_offset_params = []
    temporal_params = []  # For dual-path temporal LSTM

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen params

        if args.use_dual_path:
            # Dual-path architecture
            if 'temporal' in name or 'onset_head' in name or 'offset_head' in name:
                temporal_params.append(param)
            elif 'harmonic' in name or 'frame_head' in name or 'f0_head' in name:
                frame_f0_params.append(param)
            else:
                # CNN encoder
                shared_backbone_params.append(param)
        else:
            # Shared-LSTM architecture
            if 'onset_head' in name or 'offset_head' in name:
                onset_offset_params.append(param)
            elif 'frame_head' in name or 'f0_head' in name:
                frame_f0_params.append(param)
            else:
                # Everything else is shared backbone (CNN, shared_lstm, shared_fc, etc.)
                shared_backbone_params.append(param)

    # Determine onset/offset LR
    onset_lr = args.lr_onset if args.lr_onset is not None else (args.lr * 0.5)

    param_groups = []
    if len(shared_backbone_params) > 0:
        param_groups.append({'params': shared_backbone_params, 'lr': args.lr, 'name': 'shared_backbone'})
    if len(frame_f0_params) > 0:
        param_groups.append({'params': frame_f0_params, 'lr': args.lr, 'name': 'frame_f0'})
    if len(onset_offset_params) > 0:
        param_groups.append({'params': onset_offset_params, 'lr': onset_lr, 'name': 'onset_offset'})
    if len(temporal_params) > 0:
        param_groups.append({'params': temporal_params, 'lr': onset_lr, 'name': 'temporal'})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=0.01
    )

    print(f"\n  Parameter groups:")
    if len(shared_backbone_params) > 0:
        print(f"    Shared backbone: {sum(p.numel() for p in shared_backbone_params):,} params, LR={args.lr:.2e}")
    if len(frame_f0_params) > 0:
        print(f"    Frame/F0 (harmonic): {sum(p.numel() for p in frame_f0_params):,} params, LR={args.lr:.2e}")
    if len(onset_offset_params) > 0:
        print(f"    Onset/Offset heads: {sum(p.numel() for p in onset_offset_params):,} params, LR={onset_lr:.2e}")
    if len(temporal_params) > 0:
        print(f"    Temporal path: {sum(p.numel() for p in temporal_params):,} params, LR={onset_lr:.2e}")
    
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
    stage1_baseline_frame_f1 = None  # For auto-rollback

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")

        # Special handling for dual-path loading Stage-1 weights
        if args.use_dual_path and args.resume_weights_only:
            # Use the model's specialized weight loading method
            load_stats = model.load_stage1_weights(
                checkpoint_path=args.resume,
                device=device,
                strict_cnn=True
            )
            stage1_baseline_frame_f1 = load_stats.get('baseline_frame_f1')
        else:
            # Standard weight loading
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

            # Load model weights with strict=False for transfer learning
            # This allows loading when architectures differ (e.g., adding onset features)
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=False
            )

            if missing_keys or unexpected_keys:
                print(f"\n⚠️  Partial weight transfer (architecture mismatch):")
                if missing_keys:
                    print(f"   Missing keys (initialized randomly): {len(missing_keys)}")
                    # Show a few examples
                    for key in list(missing_keys)[:5]:
                        print(f"     - {key}")
                    if len(missing_keys) > 5:
                        print(f"     ... and {len(missing_keys) - 5} more")
                if unexpected_keys:
                    print(f"   Unexpected keys (ignored): {len(unexpected_keys)}")
                    for key in list(unexpected_keys)[:5]:
                        print(f"     - {key}")
                    if len(unexpected_keys) > 5:
                        print(f"     ... and {len(unexpected_keys) - 5} more")
            else:
                print(f"✅ All weights loaded successfully (exact architecture match)")

        if args.resume_weights_only:
            print(f"\n✅ Transfer learning mode:")
            print(f"   - Model weights: loaded from checkpoint")
            print(f"   - Optimizer: initialized fresh")
            print(f"   - Scheduler: initialized fresh")
            # For Stage 2 training, record baseline Frame F1 for auto-rollback
            # Note: For dual-path, baseline_frame_f1 is already set from load_stage1_weights()
            if not (args.use_dual_path and args.resume_weights_only):
                # Standard checkpoint loading path
                if 'val_metrics' in checkpoint and 'frame_f1' in checkpoint['val_metrics']:
                    stage1_baseline_frame_f1 = checkpoint['val_metrics']['frame_f1']
                    print(f"   - Stage 1 baseline Frame F1: {stage1_baseline_frame_f1:.4f}")
            elif stage1_baseline_frame_f1 is not None:
                print(f"   - Stage 1 baseline Frame F1: {stage1_baseline_frame_f1:.4f}")
        else:
            # Full resume (optimizer + scheduler state)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint.get('best_f1', 0.0)
            print(f"✅ Full resume from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_frame_f1': [],
        'val_frame_f1': [],
        'train_onset_f1': [],
        'val_onset_f1': [],
        'lr': [],
        'onset_weight': []  # Track onset weight warmup
    }

    # Stage 2 specific tracking
    best_frame_f1 = 0.0
    best_onset_f1 = 0.0
    rollback_checkpoint_path = None

    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")

    if args.freeze_backbone:
        print(f"\n🎯 Stage 2 Heads-Only Training:")
        print(f"   Onset weight warmup: {args.onset_warmup_schedule}")
        print(f"   Warmup over {args.onset_warmup_epochs} epochs")
        if args.auto_rollback and stage1_baseline_frame_f1 is not None:
            print(f"   Auto-rollback enabled if Frame F1 drops >{args.rollback_threshold*100:.1f}% from baseline")
            print(f"   Grace period: {args.rollback_grace_epochs} epochs (no rollback check)")
            print(f"   After rollback: {'TERMINATE' if args.rollback_terminate else 'CONTINUE with restored weights'}")
            print(f"   Baseline Frame F1: {stage1_baseline_frame_f1:.4f}")

    epochs_no_improve = 0
    loss_log_file = checkpoint_dir / "per_batch_losses.log"

    # Create log file
    with open(loss_log_file, 'w') as f:
        f.write("# Per-batch loss log\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Stage 2: Onset weight warmup schedule
        if args.freeze_backbone and hasattr(criterion, 'onset_weight'):
            warmup_progress = min(1.0, epoch / max(1, args.onset_warmup_epochs))
            if warmup_progress < 0.33:
                current_onset_weight = args.onset_warmup_schedule[0]
            elif warmup_progress < 0.67:
                current_onset_weight = args.onset_warmup_schedule[1]
            else:
                current_onset_weight = args.onset_warmup_schedule[2]

            criterion.onset_weight = current_onset_weight
            if hasattr(criterion, 'offset_weight'):
                criterion.offset_weight = current_onset_weight  # Same warmup for offset

            if epoch == 0 or epoch in [args.onset_warmup_epochs // 3, 2 * args.onset_warmup_epochs // 3]:
                print(f"\n📊 Onset weight: {current_onset_weight} (warmup progress: {warmup_progress*100:.1f}%)\n")

        # Warmup scheduler: set frame_weight based on current epoch
        if not args.no_warmup and hasattr(criterion, 'set_frame_weight'):
            if epoch < args.warmup_epochs:
                # During warmup: frame_weight = 0.0 (onset head learns without frame interference)
                criterion.set_frame_weight(0.0)
                if epoch == 0:
                    print(f"\n🔥 WARMUP MODE: frame_weight=0.0 for first {args.warmup_epochs} epochs")
                    print(f"   This lets onset head learn without frame head interference\n")
            else:
                # After warmup: restore original frame_weight
                criterion.set_frame_weight(criterion.initial_frame_weight)
                if epoch == args.warmup_epochs:
                    print(f"\n✅ WARMUP COMPLETE: Restoring frame_weight={criterion.initial_frame_weight}\n")

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
            log_file=loss_log_file,
            grad_clip=args.grad_clip,
            monitor_gradients=args.freeze_backbone  # Monitor gradients in Stage 2
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

        # Stage 2: Auto-rollback check (with grace period)
        if args.auto_rollback and stage1_baseline_frame_f1 is not None:
            # Skip rollback check during grace period
            if epoch < args.rollback_grace_epochs:
                if epoch == 0:
                    print(f"\n📊 Rollback check skipped (grace period: epoch {epoch+1}/{args.rollback_grace_epochs})")
                    print(f"   Current Frame F1: {val_metrics['frame_f1']:.4f} (baseline: {stage1_baseline_frame_f1:.4f})")
            else:
                frame_f1_drop = stage1_baseline_frame_f1 - val_metrics['frame_f1']
                if frame_f1_drop > args.rollback_threshold:
                    print(f"\n⚠️ AUTO-ROLLBACK TRIGGERED!")
                    print(f"   Frame F1 dropped {frame_f1_drop*100:.2f}% from baseline")
                    print(f"   Baseline: {stage1_baseline_frame_f1:.4f}, Current: {val_metrics['frame_f1']:.4f}")
                    if rollback_checkpoint_path and rollback_checkpoint_path.exists():
                        print(f"   Restoring from: {rollback_checkpoint_path}")
                        rollback_ckpt = torch.load(rollback_checkpoint_path, map_location=device, weights_only=False)
                        model.load_state_dict(rollback_ckpt['model_state_dict'])
                        optimizer.load_state_dict(rollback_ckpt['optimizer_state_dict'])
                        scheduler.load_state_dict(rollback_ckpt['scheduler_state_dict'])
                        print(f"   ✅ Restored checkpoint from epoch {rollback_ckpt['epoch']}")

                        if args.rollback_terminate:
                            print(f"   Terminating training to prevent further degradation")
                            break
                        else:
                            print(f"   Continuing training with restored weights...")
                            # Reset patience counter after rollback
                            epochs_no_improve = 0
                    else:
                        print(f"   ⚠️ No rollback checkpoint available, continuing...")
                elif epoch == args.rollback_grace_epochs:
                    # First check after grace period
                    print(f"\n📊 First rollback check (after grace period)")
                    print(f"   Current Frame F1: {val_metrics['frame_f1']:.4f} (baseline: {stage1_baseline_frame_f1:.4f})")
                    print(f"   Drop: {frame_f1_drop*100:.2f}% (threshold: {args.rollback_threshold*100:.1f}%)")
                    if frame_f1_drop <= args.rollback_threshold:
                        print(f"   ✅ Within acceptable range")

        # Update history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['train_frame_f1'].append(train_metrics['frame_f1'])
        history['val_frame_f1'].append(val_metrics['frame_f1'])
        history['train_onset_f1'].append(train_metrics['onset_f1'])
        history['val_onset_f1'].append(val_metrics['onset_f1'])
        history['lr'].append(current_lr)
        if args.freeze_backbone:
            history['onset_weight'].append(criterion.onset_weight if hasattr(criterion, 'onset_weight') else 0)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'best_f1': best_f1,
            'best_frame_f1': best_frame_f1,
            'best_onset_f1': best_onset_f1
        }

        # Stage 2: Save dual checkpoints (best by frame F1 and best by onset F1)
        if args.freeze_backbone:
            # Save best by Frame F1
            if val_metrics['frame_f1'] > best_frame_f1:
                best_frame_f1 = val_metrics['frame_f1']
                torch.save(checkpoint, checkpoint_dir / "best_by_frame_f1.pth")
                print(f"\n✅ New best Frame F1: {best_frame_f1:.4f}")
                # Update rollback checkpoint
                rollback_checkpoint_path = checkpoint_dir / "best_by_frame_f1.pth"

            # Save best by Onset F1
            if val_metrics['onset_f1'] > best_onset_f1:
                best_onset_f1 = val_metrics['onset_f1']
                torch.save(checkpoint, checkpoint_dir / "best_by_onset_f1.pth")
                print(f"\n✅ New best Onset F1: {best_onset_f1:.4f}")

            # Combined metric for early stopping
            combined_f1 = (val_metrics['frame_f1'] + val_metrics['onset_f1']) / 2
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            # Stage 1: Standard combined F1
            combined_f1 = (val_metrics['frame_f1'] + val_metrics['onset_f1']) / 2

            # Save best model
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                torch.save(checkpoint, checkpoint_dir / "best_model.pth")
                print(f"\n✅ New best F1: {best_f1:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        # Save epoch checkpoint
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch + 1}.pth")
        
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
    print(f"Logs saved to: {loss_log_file}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
