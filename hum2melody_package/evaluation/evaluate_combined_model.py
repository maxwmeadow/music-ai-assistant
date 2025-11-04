#!/usr/bin/env python3
"""
Evaluate Combined Model

Comprehensive evaluation of the combined pitch + onset/offset model.

Measures:
- Pitch metrics: Frame F1, accuracy, within-N-semitone
- Onset/Offset metrics: F1, precision, recall
- Practical metrics: Contour accuracy, octave errors

Usage:
    python evaluate_combined_model.py \\
        --checkpoint combined_hum2melody_full.pth \\
        --manifest dataset/combined_manifest.json \\
        --output combined_evaluation.json
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.combined_model_loader import load_combined_model
from data.melody_dataset import EnhancedMelodyDataset
from torch.utils.data import DataLoader, random_split


def collate_fn(batch):
    """Custom collate for (features, targets) format."""
    features_list = []
    targets_list = defaultdict(list)

    for features, targets in batch:
        features_list.append(features)
        for key, value in targets.items():
            targets_list[key].append(value)

    batched_features = torch.stack(features_list, dim=0)
    batched_targets = {key: torch.stack(values, dim=0)
                      for key, values in targets_list.items()}

    return batched_features, batched_targets


def interpolate_target(
    target: torch.Tensor,
    target_name: str,
    src_frames: int,
    dst_frames: int,
    device: torch.device
) -> torch.Tensor:
    """Resample target from src_frames -> dst_frames."""
    t = target.to(device).float()

    if t.dim() == 2:
        t = t.unsqueeze(-1)

    batch, s_frames, channels = t.shape
    t = t.permute(0, 2, 1)  # (batch, channels, time)
    t = t.unsqueeze(-1)  # (batch, channels, time, 1)

    # Choose interpolation mode
    if target_name in ['onset', 'offset']:
        interpolation_mode = 'nearest'
        is_binary = True
    elif target_name == 'frame':
        interpolation_mode = 'bilinear'
        is_binary = True
    else:  # f0
        interpolation_mode = 'bilinear'
        is_binary = False

    interpolated = F.interpolate(
        t, size=(dst_frames, 1), mode=interpolation_mode,
        align_corners=False if interpolation_mode == 'bilinear' else None
    )
    interpolated = interpolated.squeeze(-1)  # (batch, channels, dst_frames)
    interpolated = interpolated.permute(0, 2, 1)  # (batch, dst_frames, channels)

    if is_binary:
        interpolated = (interpolated > 0.5).float()
        if channels == 1:
            interpolated = interpolated.squeeze(-1)

    return interpolated


def compute_metrics(outputs_tuple, targets, device):
    """
    Compute metrics from combined model outputs.

    Args:
        outputs_tuple: Tuple of (frame, onset, offset, f0) tensors
        targets: Dict of target tensors
        device: Device to use for computation

    Returns:
        Dict of metric values (raw counts for aggregation)
    """
    frame_pred, onset_pred, offset_pred, f0_pred = outputs_tuple
    out = {}

    # --- PITCH METRICS ---
    frame_targ = targets.get('frame', None)

    if frame_pred is not None and frame_targ is not None:
        fp = frame_pred.detach().cpu()
        ft = frame_targ.detach().cpu()

        # Resample targets if needed
        output_frames = fp.shape[1]
        target_frames = ft.shape[0] if ft.dim() == 2 else ft.shape[1]

        if output_frames != target_frames:
            ft_gpu = ft.to(device)
            if ft_gpu.dim() == 2:
                ft_gpu = ft_gpu.unsqueeze(0)
            ft_resampled = interpolate_target(ft_gpu, 'frame', target_frames, output_frames, device)
            ft = ft_resampled.cpu()

        # Ensure consistent dimensions
        if fp.dim() == 2:
            fp = fp.unsqueeze(-1)
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1)

        # Detect if logits or probs
        is_logits = (fp.max() > 2.0) or (fp.min() < -2.0)
        prob = torch.sigmoid(fp) if is_logits else fp.clamp(0.0, 1.0)

        # Frame F1
        pred_bin = (prob > 0.5).float()
        p_flat = pred_bin.numpy().reshape(-1)
        t_flat = (ft.numpy().reshape(-1) > 0.5).astype(float)

        tp = float(((p_flat == 1) & (t_flat == 1)).sum())
        fp_cnt = float(((p_flat == 1) & (t_flat == 0)).sum())
        fn = float(((p_flat == 0) & (t_flat == 1)).sum())

        out['frame_tp'] = tp
        out['frame_fp'] = fp_cnt
        out['frame_fn'] = fn

        # Pitch accuracy
        pred_idx = torch.argmax(prob, dim=-1).numpy()
        targ_idx = torch.argmax(ft, dim=-1).numpy()
        correct = (pred_idx == targ_idx).astype(float)
        out['pitch_correct'] = float(correct.sum())
        out['pitch_total'] = float(correct.size)

        # Voiced frame metrics
        voiced_mask = (ft.sum(dim=-1) > 0.5).numpy()
        if voiced_mask.sum() > 0:
            pred_voiced = pred_idx[voiced_mask]
            targ_voiced = targ_idx[voiced_mask]

            signed_error = (pred_voiced - targ_voiced).astype(int)
            pitch_error = np.abs(signed_error)

            out['exact_voiced_correct'] = float((pitch_error == 0).sum())
            out['within_1st_correct'] = float((pitch_error <= 1).sum())
            out['within_2st_correct'] = float((pitch_error <= 2).sum())
            out['within_3st_correct'] = float((pitch_error <= 3).sum())
            out['error_sum'] = float(pitch_error.sum())
            out['error_list'] = pitch_error.tolist()
            out['signed_error_list'] = signed_error.tolist()

            # Octave-invariant
            pred_chroma = pred_voiced % 12
            targ_chroma = targ_voiced % 12
            out['octave_correct'] = float((pred_chroma == targ_chroma).sum())

            # Pitch contour
            if len(pred_voiced) > 1:
                pred_diff = np.sign(pred_voiced[1:] - pred_voiced[:-1])
                targ_diff = np.sign(targ_voiced[1:] - targ_voiced[:-1])
                out['contour_correct'] = float((pred_diff == targ_diff).sum())
                out['contour_total'] = float(len(pred_diff))
            else:
                out['contour_correct'] = 0.0
                out['contour_total'] = 0.0

            out['voiced_frames'] = int(voiced_mask.sum())
        else:
            out.update({
                'exact_voiced_correct': 0.0,
                'within_1st_correct': 0.0,
                'within_2st_correct': 0.0,
                'within_3st_correct': 0.0,
                'error_sum': 0.0,
                'error_list': [],
                'signed_error_list': [],
                'octave_correct': 0.0,
                'contour_correct': 0.0,
                'contour_total': 0.0,
                'voiced_frames': 0
            })

        out['total_frames'] = pred_idx.size
    else:
        # No pitch metrics
        out.update({
            'frame_tp': 0.0, 'frame_fp': 0.0, 'frame_fn': 0.0,
            'pitch_correct': 0.0, 'pitch_total': 0.0,
            'exact_voiced_correct': 0.0, 'within_1st_correct': 0.0,
            'within_2st_correct': 0.0, 'within_3st_correct': 0.0,
            'error_sum': 0.0, 'error_list': [],
            'signed_error_list': [], 'octave_correct': 0.0,
            'contour_correct': 0.0, 'contour_total': 0.0,
            'voiced_frames': 0, 'total_frames': 0
        })

    # --- ONSET METRICS ---
    onset_targ = targets.get('onset', None)

    if onset_pred is not None and onset_targ is not None:
        op = onset_pred.detach().cpu()
        ot = onset_targ.detach().cpu()

        # Resample if needed
        output_frames = op.shape[1] if op.dim() >= 2 else op.shape[0]
        # FIX: After batching, onset targets are (batch, time), so get time from shape[1]
        target_frames = ot.shape[1] if ot.dim() >= 2 else ot.shape[0]

        if output_frames != target_frames:
            ot_gpu = ot.to(device)
            ot_resampled = interpolate_target(ot_gpu, 'onset', target_frames, output_frames, device)
            ot = ot_resampled.cpu()

        # Onset/offset are already binary from dataset (not multi-hot like frame)
        # Only convert if it's actually multi-hot (3D)
        if ot.dim() == 3:
            ot = (ot.sum(dim=-1) > 0.5).float()

        while op.dim() > 2:
            op = op.squeeze(-1)

        # Get probs
        is_logits = (op.max() > 2.0) or (op.min() < -2.0)
        onset_prob = torch.sigmoid(op) if is_logits else op.clamp(0.0, 1.0)

        pred_bin = (onset_prob > 0.5).float()

        if pred_bin.dim() == 1:
            pred_bin = pred_bin.unsqueeze(0)
        if ot.dim() == 1:
            ot = ot.unsqueeze(0)

        p_flat = pred_bin.numpy().reshape(-1)
        t_flat = ot.numpy().reshape(-1)

        if len(p_flat) == len(t_flat):
            onset_tp = float(((p_flat == 1) & (t_flat == 1)).sum())
            onset_fp = float(((p_flat == 1) & (t_flat == 0)).sum())
            onset_fn = float(((p_flat == 0) & (t_flat == 1)).sum())
            out['onset_tp'] = onset_tp
            out['onset_fp'] = onset_fp
            out['onset_fn'] = onset_fn
        else:
            print(f"WARNING: onset shape mismatch: pred={pred_bin.shape}, target={ot.shape}")
            out.update({'onset_tp': 0.0, 'onset_fp': 0.0, 'onset_fn': 0.0})
    else:
        out.update({'onset_tp': 0.0, 'onset_fp': 0.0, 'onset_fn': 0.0})

    # --- OFFSET METRICS ---
    offset_targ = targets.get('offset', None)

    if offset_pred is not None and offset_targ is not None:
        ofp = offset_pred.detach().cpu()
        oft = offset_targ.detach().cpu()

        # Resample if needed
        output_frames = ofp.shape[1] if ofp.dim() >= 2 else ofp.shape[0]
        # FIX: After batching, offset targets are (batch, time), so get time from shape[1]
        target_frames = oft.shape[1] if oft.dim() >= 2 else oft.shape[0]

        if output_frames != target_frames:
            oft_gpu = oft.to(device)
            oft_resampled = interpolate_target(oft_gpu, 'offset', target_frames, output_frames, device)
            oft = oft_resampled.cpu()

        # Onset/offset are already binary from dataset (not multi-hot like frame)
        # Only convert if it's actually multi-hot (3D)
        if oft.dim() == 3:
            oft = (oft.sum(dim=-1) > 0.5).float()

        while ofp.dim() > 2:
            ofp = ofp.squeeze(-1)

        # Get probs
        is_logits = (ofp.max() > 2.0) or (ofp.min() < -2.0)
        offset_prob = torch.sigmoid(ofp) if is_logits else ofp.clamp(0.0, 1.0)

        pred_bin = (offset_prob > 0.5).float()

        if pred_bin.dim() == 1:
            pred_bin = pred_bin.unsqueeze(0)
        if oft.dim() == 1:
            oft = oft.unsqueeze(0)

        p_flat = pred_bin.numpy().reshape(-1)
        t_flat = oft.numpy().reshape(-1)

        if len(p_flat) == len(t_flat):
            offset_tp = float(((p_flat == 1) & (t_flat == 1)).sum())
            offset_fp = float(((p_flat == 1) & (t_flat == 0)).sum())
            offset_fn = float(((p_flat == 0) & (t_flat == 1)).sum())
            out['offset_tp'] = offset_tp
            out['offset_fp'] = offset_fp
            out['offset_fn'] = offset_fn
        else:
            print(f"WARNING: offset shape mismatch: pred={pred_bin.shape}, target={oft.shape}")
            out.update({'offset_tp': 0.0, 'offset_fp': 0.0, 'offset_fn': 0.0})
    else:
        out.update({'offset_tp': 0.0, 'offset_fp': 0.0, 'offset_fn': 0.0})

    return out


def evaluate_model(model, dataloader, device):
    """Evaluate combined model and aggregate metrics."""
    model.eval()
    all_metrics = defaultdict(list)

    with torch.no_grad():
        for features_batch, targets_batch in tqdm(dataloader, desc="Evaluating"):
            features = features_batch.to(device)
            targets = {k: v.to(device) for k, v in targets_batch.items()}

            # Split features into CQT and extras
            # Input: (batch, 1, time, 112)
            cqt = features[:, :, :, :88]
            extras = features[:, :, :, 88:]

            # Forward pass - combined model returns tuple
            outputs_tuple = model(cqt, extras)

            # Compute metrics
            metrics = compute_metrics(outputs_tuple, targets, device)

            # Store
            for k, v in metrics.items():
                if isinstance(v, list):
                    all_metrics[k].append(v)
                elif isinstance(v, (int, float)) and not np.isnan(v):
                    all_metrics[k].append(v)

    # Aggregate
    aggregated = {}

    count_metrics = ['frame_tp', 'frame_fp', 'frame_fn', 'pitch_correct', 'pitch_total',
                     'exact_voiced_correct', 'within_1st_correct', 'within_2st_correct',
                     'within_3st_correct', 'error_sum', 'octave_correct',
                     'contour_correct', 'contour_total', 'voiced_frames', 'total_frames',
                     'onset_tp', 'onset_fp', 'onset_fn', 'offset_tp', 'offset_fp', 'offset_fn']

    for k in count_metrics:
        if k in all_metrics:
            aggregated[k] = sum(all_metrics[k])

    # Error lists
    if 'error_list' in all_metrics:
        all_errors = []
        for batch_errors in all_metrics['error_list']:
            all_errors.extend(batch_errors)
        aggregated['median_error'] = float(np.median(all_errors)) if all_errors else 0.0
        aggregated['all_errors'] = all_errors

    if 'signed_error_list' in all_metrics:
        all_signed_errors = []
        for batch_errors in all_metrics['signed_error_list']:
            all_signed_errors.extend(batch_errors)
        aggregated['all_signed_errors'] = all_signed_errors

    # Compute metrics from counts
    # Frame F1
    tp = aggregated.get('frame_tp', 0)
    fp = aggregated.get('frame_fp', 0)
    fn = aggregated.get('frame_fn', 0)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    aggregated['frame_f1'] = 2 * prec * rec / (prec + rec + 1e-9)

    # Pitch accuracy
    aggregated['pitch_acc'] = aggregated.get('pitch_correct', 0) / max(aggregated.get('pitch_total', 1), 1)

    # Voiced metrics
    voiced = max(aggregated.get('voiced_frames', 1), 1)
    aggregated['exact_voiced_acc'] = aggregated.get('exact_voiced_correct', 0) / voiced
    aggregated['within_1st'] = aggregated.get('within_1st_correct', 0) / voiced
    aggregated['within_2st'] = aggregated.get('within_2st_correct', 0) / voiced
    aggregated['within_3st'] = aggregated.get('within_3st_correct', 0) / voiced
    aggregated['mean_error_semitones'] = aggregated.get('error_sum', 0) / voiced
    aggregated['median_error_semitones'] = aggregated.get('median_error', 0)
    aggregated['octave_acc'] = aggregated.get('octave_correct', 0) / voiced

    # Contour
    contour_total = max(aggregated.get('contour_total', 1), 1)
    aggregated['contour_acc'] = aggregated.get('contour_correct', 0) / contour_total

    # Onset F1
    onset_tp = aggregated.get('onset_tp', 0)
    onset_fp = aggregated.get('onset_fp', 0)
    onset_fn = aggregated.get('onset_fn', 0)
    onset_prec = onset_tp / (onset_tp + onset_fp + 1e-9)
    onset_rec = onset_tp / (onset_tp + onset_fn + 1e-9)
    aggregated['onset_f1'] = 2 * onset_prec * onset_rec / (onset_prec + onset_rec + 1e-9)

    # Offset F1
    offset_tp = aggregated.get('offset_tp', 0)
    offset_fp = aggregated.get('offset_fp', 0)
    offset_fn = aggregated.get('offset_fn', 0)
    offset_prec = offset_tp / (offset_tp + offset_fp + 1e-9)
    offset_rec = offset_tp / (offset_tp + offset_fn + 1e-9)
    aggregated['offset_f1'] = 2 * offset_prec * offset_rec / (offset_prec + offset_rec + 1e-9)

    return aggregated


def analyze_error_distribution(signed_errors):
    """Analyze signed error distribution."""
    if not signed_errors:
        return None

    errors_clipped = np.clip(signed_errors, -48, 48).astype(int)
    counter = Counter(errors_clipped)
    total = len(signed_errors)

    top_errors = []
    for error_val, count in counter.most_common(20):
        pct = 100.0 * count / total
        top_errors.append({
            'error_semitones': int(error_val),
            'count': int(count),
            'percentage': float(pct)
        })

    octave_multiples = [12, -12, 24, -24, 36, -36]
    octave_error_count = sum(counter.get(mult, 0) for mult in octave_multiples)
    octave_error_pct = 100.0 * octave_error_count / total

    abs_errors = np.abs(errors_clipped)
    perfect = int((abs_errors == 0).sum())
    within_1 = int((abs_errors <= 1).sum())
    within_3 = int((abs_errors <= 3).sum())
    within_6 = int((abs_errors <= 6).sum())
    large = int((abs_errors > 6).sum())

    return {
        'histogram': dict(counter),
        'top_errors': top_errors,
        'octave_analysis': {
            'octave_multiple_count': int(octave_error_count),
            'octave_multiple_percentage': float(octave_error_pct),
            'is_dominant_pattern': octave_error_pct > 15.0
        },
        'badness_categories': {
            'perfect': {'count': perfect, 'pct': 100.0 * perfect / total},
            'within_1st': {'count': within_1, 'pct': 100.0 * within_1 / total},
            'within_3st': {'count': within_3, 'pct': 100.0 * within_3 / total},
            'within_6st': {'count': within_6, 'pct': 100.0 * within_6 / total},
            'large_errors': {'count': large, 'pct': 100.0 * large / total}
        },
        'total_frames': total
    }


def print_results(metrics, checkpoint_name, error_analysis=None):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print(f"COMBINED MODEL EVALUATION: {checkpoint_name}")
    print("=" * 70)

    print("\n📊 PITCH METRICS")
    print("-" * 70)
    print(f"  Frame F1 (multi-label):      {metrics['frame_f1']*100:6.2f}%")
    print(f"  Pitch Accuracy (argmax):     {metrics['pitch_acc']*100:6.2f}%")
    print(f"  Exact Match (voiced):        {metrics['exact_voiced_acc']*100:6.2f}%")
    print(f"  Within ±1 Semitone:          {metrics['within_1st']*100:6.2f}%  ⭐")
    print(f"  Within ±2 Semitones:         {metrics['within_2st']*100:6.2f}%")
    print(f"  Within ±3 Semitones:         {metrics['within_3st']*100:6.2f}%")
    print(f"  Octave-Invariant:            {metrics['octave_acc']*100:6.2f}%")
    print(f"  Pitch Contour (direction):   {metrics['contour_acc']*100:6.2f}%  ⭐")

    print("\n📊 ONSET/OFFSET METRICS")
    print("-" * 70)
    print(f"  Onset F1:                    {metrics.get('onset_f1', 0)*100:6.2f}%")
    print(f"  Offset F1:                   {metrics.get('offset_f1', 0)*100:6.2f}%")

    print("\n📉 ERROR STATISTICS")
    print("-" * 70)
    print(f"  Mean Error:                  {metrics['mean_error_semitones']:.2f} semitones")
    print(f"  Median Error:                {metrics['median_error_semitones']:.2f} semitones")

    if error_analysis:
        print("\n🔍 ERROR DISTRIBUTION ANALYSIS")
        print("-" * 70)

        bad = error_analysis['badness_categories']
        print(f"  Perfect (0 error):           {bad['perfect']['pct']:6.2f}% ({bad['perfect']['count']:,} frames)")
        print(f"  Within ±1 semitone:          {bad['within_1st']['pct']:6.2f}% ({bad['within_1st']['count']:,} frames)")
        print(f"  Within ±3 semitones:         {bad['within_3st']['pct']:6.2f}% ({bad['within_3st']['count']:,} frames)")
        print(f"  Within ±6 semitones:         {bad['within_6st']['pct']:6.2f}% ({bad['within_6st']['count']:,} frames)")
        print(f"  Large errors (>6 st):        {bad['large_errors']['pct']:6.2f}% ({bad['large_errors']['count']:,} frames)")

        oct = error_analysis['octave_analysis']
        print(f"\n  Octave-multiple errors:")
        print(f"    Count (±12, ±24, ±36):     {oct['octave_multiple_count']:,} frames")
        print(f"    Percentage:                {oct['octave_multiple_percentage']:.2f}%")
        if oct['is_dominant_pattern']:
            print(f"    ⚠️  OCTAVE ERRORS ARE DOMINANT!")
        else:
            print(f"    ✓ Octave errors are not the main issue")

        print(f"\n  Top 10 most common errors:")
        for i, err in enumerate(error_analysis['top_errors'][:10], 1):
            marker = " ← OCTAVE" if abs(err['error_semitones']) in [12, 24, 36] else ""
            print(f"    {i:2d}. {err['error_semitones']:+4d} semitones: {err['percentage']:5.2f}% ({err['count']:,} frames){marker}")

    print("\n📊 FRAME COUNTS")
    print("-" * 70)
    print(f"  Voiced Frames:               {metrics['voiced_frames']:,}")
    print(f"  Total Frames:                {metrics['total_frames']:,}")
    print(f"  Voiced Ratio:                {metrics['voiced_frames']/max(metrics['total_frames'], 1)*100:.1f}%")

    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    within_1 = metrics['within_1st'] * 100
    contour = metrics['contour_acc'] * 100
    onset_f1 = metrics.get('onset_f1', 0) * 100
    offset_f1 = metrics.get('offset_f1', 0) * 100

    if within_1 >= 85 and contour >= 80 and onset_f1 >= 25:
        print("✅ EXCELLENT - Ready for production!")
        print(f"   Pitch: Within-1-semitone ({within_1:.1f}%) is great")
        print(f"   Pitch: Contour ({contour:.1f}%) preserves melody shape")
        print(f"   Onset: F1 ({onset_f1:.1f}%) meets target")
        print(f"   Offset: F1 ({offset_f1:.1f}%) meets target")
    elif within_1 >= 75 and onset_f1 >= 20:
        print("✅ GOOD - Usable for melody transcription")
        print(f"   Pitch metrics are good")
        print(f"   Onset/Offset detection is functional")
    else:
        print("⚠️  NEEDS IMPROVEMENT")
        print(f"   Some metrics below target")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to combined checkpoint')
    parser.add_argument('--manifest', type=str,
                       default='dataset/combined_manifest.json',
                       help='Path to dataset manifest')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (use 0 to avoid cffi issues)')
    parser.add_argument('--output', type=str, default='combined_evaluation.json',
                       help='Path to save results JSON')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1 = 10%)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load combined model
    print(f"\nLoading combined model from: {args.checkpoint}")
    model = load_combined_model(args.checkpoint, device=str(device))

    # Create dataset
    print(f"\nLoading dataset from: {args.manifest}")
    full_dataset = EnhancedMelodyDataset(
        labels_path=args.manifest,
        n_bins=88,
        augment=False,
        spec_augment=False,
        use_onset_features=True,
        use_musical_context=True
    )
    print(f"Full dataset: {len(full_dataset)} samples")

    # Split into train/val (use validation set for evaluation)
    train_size = int((1.0 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Using validation set: {val_size} samples ({args.val_split*100:.0f}%)")

    # Create dataloader
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate_model(model, dataloader, device)

    # Analyze errors
    error_analysis = None
    if 'all_signed_errors' in metrics:
        print("\nAnalyzing error distribution...")
        error_analysis = analyze_error_distribution(metrics['all_signed_errors'])

    # Print results
    checkpoint_name = Path(args.checkpoint).stem
    print_results(metrics, checkpoint_name, error_analysis)

    # Save results
    metrics_for_json = {k: v for k, v in metrics.items()
                        if k not in ['all_errors', 'all_signed_errors']}

    output_data = {
        'checkpoint': args.checkpoint,
        'metrics': metrics_for_json,
        'val_samples': val_size,
        'val_split': args.val_split
    }

    if error_analysis:
        error_analysis_summary = {
            'top_errors': error_analysis['top_errors'],
            'octave_analysis': error_analysis['octave_analysis'],
            'badness_categories': error_analysis['badness_categories'],
            'total_frames': error_analysis['total_frames']
        }
        output_data['error_analysis'] = error_analysis_summary

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
