#!/usr/bin/env python3
"""
Enhanced Combined Model Evaluation with Event-Based Onset/Offset Metrics

Adds sophisticated onset/offset evaluation:
1. Frame-level F1 (exact matching)
2. Event-based F1 with temporal tolerance (±25ms, ±50ms, ±100ms)
3. Timing error statistics (mean/median offset from true boundaries)
4. Detection rate analysis (how many notes are captured)

This evaluates how well the model detects actual note events, not just frame accuracy.

Usage:
    python evaluate_combined_detailed.py \\
        --checkpoint combined_hum2melody_full.pth \\
        --manifest dataset/combined_manifest.json \\
        --output combined_detailed_evaluation.json
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d
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


def interpolate_target(target, target_name, src_frames, dst_frames, device):
    """Resample target from src_frames -> dst_frames."""
    t = target.to(device).float()

    if t.dim() == 2:
        t = t.unsqueeze(-1)

    batch, s_frames, channels = t.shape
    t = t.permute(0, 2, 1)
    t = t.unsqueeze(-1)

    if target_name in ['onset', 'offset']:
        interpolation_mode = 'nearest'
        is_binary = True
    elif target_name == 'frame':
        interpolation_mode = 'bilinear'
        is_binary = True
    else:
        interpolation_mode = 'bilinear'
        is_binary = False

    interpolated = F.interpolate(
        t, size=(dst_frames, 1), mode=interpolation_mode,
        align_corners=False if interpolation_mode == 'bilinear' else None
    )
    interpolated = interpolated.squeeze(-1)
    interpolated = interpolated.permute(0, 2, 1)

    if is_binary:
        interpolated = (interpolated > 0.5).float()
        if channels == 1:
            interpolated = interpolated.squeeze(-1)

    return interpolated


def extract_events_from_frames(frame_probs, threshold=0.5, min_separation_frames=2):
    """
    Extract onset/offset events from frame-level predictions.

    Args:
        frame_probs: (time,) - probability of onset/offset at each frame
        threshold: Detection threshold
        min_separation_frames: Minimum frames between events

    Returns:
        event_frames: List of frame indices where events occur
    """
    # Find peaks above threshold
    peaks, properties = find_peaks(
        frame_probs,
        height=threshold,
        distance=min_separation_frames
    )

    return peaks.tolist()


def match_events_with_tolerance(pred_events, true_events, tolerance_frames):
    """
    Match predicted events to ground truth with temporal tolerance.

    Uses greedy matching: each prediction matched to nearest ground truth within tolerance.

    Args:
        pred_events: List of predicted event frames
        true_events: List of ground truth event frames
        tolerance_frames: Temporal tolerance window (±N frames)

    Returns:
        dict with:
        - tp: True positives (matched predictions)
        - fp: False positives (unmatched predictions)
        - fn: False negatives (unmatched ground truth)
        - timing_errors: List of timing errors for matched events (frames)
    """
    pred_events = sorted(pred_events)
    true_events = sorted(true_events)

    if len(pred_events) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': len(true_events),
            'timing_errors': []
        }

    if len(true_events) == 0:
        return {
            'tp': 0,
            'fp': len(pred_events),
            'fn': 0,
            'timing_errors': []
        }

    # Track which true events have been matched
    true_matched = [False] * len(true_events)
    timing_errors = []
    tp = 0

    # For each prediction, find closest true event within tolerance
    for pred_frame in pred_events:
        best_match_idx = None
        best_distance = float('inf')

        for true_idx, true_frame in enumerate(true_events):
            if true_matched[true_idx]:
                continue

            distance = abs(pred_frame - true_frame)
            if distance <= tolerance_frames and distance < best_distance:
                best_match_idx = true_idx
                best_distance = distance

        if best_match_idx is not None:
            # Match found
            true_matched[best_match_idx] = True
            tp += 1
            timing_errors.append(pred_events[pred_events.index(pred_frame)] - true_events[best_match_idx])

    fp = len(pred_events) - tp
    fn = len(true_events) - tp

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'timing_errors': timing_errors
    }


def compute_event_based_metrics(pred_probs, true_labels, frame_rate=31.25, threshold=0.5):
    """
    Compute event-based onset/offset metrics at multiple tolerance levels.

    Args:
        pred_probs: (batch, time) - prediction probabilities
        true_labels: (batch, time) - ground truth binary labels
        frame_rate: Frames per second
        threshold: Detection threshold for predictions

    Returns:
        dict with metrics at different tolerance levels
    """
    results = {
        'frame_level': {'tp': 0, 'fp': 0, 'fn': 0},
        'event_25ms': {'tp': 0, 'fp': 0, 'fn': 0, 'timing_errors': []},
        'event_50ms': {'tp': 0, 'fp': 0, 'fn': 0, 'timing_errors': []},
        'event_100ms': {'tp': 0, 'fp': 0, 'fn': 0, 'timing_errors': []},
        'pred_event_count': 0,
        'true_event_count': 0
    }

    # Tolerance levels in frames
    tol_25ms = int(0.025 * frame_rate)  # ±25ms
    tol_50ms = int(0.050 * frame_rate)  # ±50ms
    tol_100ms = int(0.100 * frame_rate)  # ±100ms

    # Process each sample in batch
    for batch_idx in range(pred_probs.shape[0]):
        pred = pred_probs[batch_idx].cpu().numpy()
        true = true_labels[batch_idx].cpu().numpy()

        # Resample targets if shape mismatch (model outputs 125 frames, dataset has 500)
        if len(pred) != len(true):
            # Use nearest interpolation to preserve binary labels
            import torch.nn.functional as F_interp
            true_tensor = torch.FloatTensor(true).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            target_frames = len(pred)
            source_frames = len(true)
            resampled = F_interp.interpolate(
                true_tensor,
                size=(target_frames, 1),
                mode='nearest'
            )
            true = resampled.squeeze().cpu().numpy()

        # Frame-level metrics (exact matching)
        pred_bin = (pred > threshold).astype(float)
        results['frame_level']['tp'] += ((pred_bin == 1) & (true == 1)).sum()
        results['frame_level']['fp'] += ((pred_bin == 1) & (true == 0)).sum()
        results['frame_level']['fn'] += ((pred_bin == 0) & (true == 1)).sum()

        # Extract events
        pred_events = extract_events_from_frames(pred, threshold=threshold)
        true_events = extract_events_from_frames(true, threshold=0.5)

        results['pred_event_count'] += len(pred_events)
        results['true_event_count'] += len(true_events)

        # Event-based metrics at different tolerances
        for tol_frames, key in [(tol_25ms, 'event_25ms'),
                                 (tol_50ms, 'event_50ms'),
                                 (tol_100ms, 'event_100ms')]:
            match_result = match_events_with_tolerance(pred_events, true_events, tol_frames)
            results[key]['tp'] += match_result['tp']
            results[key]['fp'] += match_result['fp']
            results[key]['fn'] += match_result['fn']
            results[key]['timing_errors'].extend(match_result['timing_errors'])

    # Compute F1 scores
    for key in ['frame_level', 'event_25ms', 'event_50ms', 'event_100ms']:
        tp = results[key]['tp']
        fp = results[key]['fp']
        fn = results[key]['fn']

        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        results[key]['precision'] = prec
        results[key]['recall'] = rec
        results[key]['f1'] = f1

    # Timing error statistics (in milliseconds)
    for key in ['event_25ms', 'event_50ms', 'event_100ms']:
        errors = results[key]['timing_errors']
        if errors:
            errors_ms = np.array(errors) * 1000.0 / frame_rate
            results[key]['mean_timing_error_ms'] = float(np.mean(np.abs(errors_ms)))
            results[key]['median_timing_error_ms'] = float(np.median(np.abs(errors_ms)))
            results[key]['std_timing_error_ms'] = float(np.std(errors_ms))
        else:
            results[key]['mean_timing_error_ms'] = 0.0
            results[key]['median_timing_error_ms'] = 0.0
            results[key]['std_timing_error_ms'] = 0.0

    return results


def evaluate_model_detailed(model, dataloader, device):
    """Evaluate combined model with detailed onset/offset analysis."""
    model.eval()

    # Accumulators for pitch metrics
    pitch_metrics = defaultdict(list)

    # Accumulators for onset/offset
    onset_predictions = []
    onset_targets = []
    offset_predictions = []
    offset_targets = []

    with torch.no_grad():
        for features_batch, targets_batch in tqdm(dataloader, desc="Evaluating"):
            features = features_batch.to(device)
            targets = {k: v.to(device) for k, v in targets_batch.items()}

            # Split features
            cqt = features[:, :, :, :88]
            extras = features[:, :, :, 88:]

            # Forward pass
            frame_pred, onset_pred, offset_pred, f0_pred = model(cqt, extras)

            # Store onset/offset for detailed analysis
            onset_predictions.append(onset_pred.squeeze(-1))
            onset_targets.append(targets['onset'])
            offset_predictions.append(offset_pred.squeeze(-1))
            offset_targets.append(targets['offset'])

            # Compute pitch metrics (same as before)
            frame_targ = targets.get('frame', None)
            if frame_pred is not None and frame_targ is not None:
                fp = frame_pred.detach().cpu()
                ft = frame_targ.detach().cpu()

                # Resample if needed
                output_frames = fp.shape[1]
                target_frames = ft.shape[1] if ft.dim() >= 2 else ft.shape[0]

                if output_frames != target_frames:
                    ft_gpu = ft.to(device)
                    if ft_gpu.dim() == 2:
                        ft_gpu = ft_gpu.unsqueeze(0)
                    ft = interpolate_target(ft_gpu, 'frame', target_frames, output_frames, device).cpu()

                if fp.dim() == 2:
                    fp = fp.unsqueeze(-1)
                if ft.dim() == 2:
                    ft = ft.unsqueeze(-1)

                is_logits = (fp.max() > 2.0) or (fp.min() < -2.0)
                prob = torch.sigmoid(fp) if is_logits else fp.clamp(0.0, 1.0)

                pred_bin = (prob > 0.5).float()
                p_flat = pred_bin.numpy().reshape(-1)
                t_flat = (ft.numpy().reshape(-1) > 0.5).astype(float)

                tp = float(((p_flat == 1) & (t_flat == 1)).sum())
                fp_cnt = float(((p_flat == 1) & (t_flat == 0)).sum())
                fn = float(((p_flat == 0) & (t_flat == 1)).sum())

                pitch_metrics['frame_tp'].append(tp)
                pitch_metrics['frame_fp'].append(fp_cnt)
                pitch_metrics['frame_fn'].append(fn)

                # Voiced frame metrics
                pred_idx = torch.argmax(prob, dim=-1).numpy()
                targ_idx = torch.argmax(ft, dim=-1).numpy()
                correct = (pred_idx == targ_idx).astype(float)
                pitch_metrics['pitch_correct'].append(float(correct.sum()))
                pitch_metrics['pitch_total'].append(float(correct.size))

                voiced_mask = (ft.sum(dim=-1) > 0.5).numpy()
                if voiced_mask.sum() > 0:
                    pred_voiced = pred_idx[voiced_mask]
                    targ_voiced = targ_idx[voiced_mask]
                    pitch_error = np.abs(pred_voiced - targ_voiced)

                    pitch_metrics['within_1st_correct'].append(float((pitch_error <= 1).sum()))
                    pitch_metrics['voiced_frames'].append(int(voiced_mask.sum()))

    # Aggregate pitch metrics
    aggregated = {}
    for k in ['frame_tp', 'frame_fp', 'frame_fn', 'pitch_correct', 'pitch_total',
              'within_1st_correct', 'voiced_frames']:
        if k in pitch_metrics:
            aggregated[k] = sum(pitch_metrics[k])

    tp = aggregated.get('frame_tp', 0)
    fp = aggregated.get('frame_fp', 0)
    fn = aggregated.get('frame_fn', 0)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    aggregated['frame_f1'] = 2 * prec * rec / (prec + rec + 1e-9)
    aggregated['pitch_acc'] = aggregated.get('pitch_correct', 0) / max(aggregated.get('pitch_total', 1), 1)
    voiced = max(aggregated.get('voiced_frames', 1), 1)
    aggregated['within_1st'] = aggregated.get('within_1st_correct', 0) / voiced

    # Compute detailed onset/offset metrics
    onset_predictions_cat = torch.cat(onset_predictions, dim=0)
    onset_targets_cat = torch.cat(onset_targets, dim=0)
    offset_predictions_cat = torch.cat(offset_predictions, dim=0)
    offset_targets_cat = torch.cat(offset_targets, dim=0)

    print("\n\nComputing detailed onset metrics...")
    onset_detailed = compute_event_based_metrics(
        torch.sigmoid(onset_predictions_cat),
        onset_targets_cat
    )

    print("Computing detailed offset metrics...")
    offset_detailed = compute_event_based_metrics(
        torch.sigmoid(offset_predictions_cat),
        offset_targets_cat
    )

    return {
        'pitch_metrics': aggregated,
        'onset_detailed': onset_detailed,
        'offset_detailed': offset_detailed
    }


def print_detailed_results(results):
    """Print detailed evaluation results."""
    print("\n" + "=" * 80)
    print("DETAILED COMBINED MODEL EVALUATION")
    print("=" * 80)

    pitch = results['pitch_metrics']
    print("\n📊 PITCH METRICS")
    print("-" * 80)
    print(f"  Frame F1:                    {pitch['frame_f1']*100:6.2f}%")
    print(f"  Pitch Accuracy:              {pitch['pitch_acc']*100:6.2f}%")
    print(f"  Within ±1 Semitone:          {pitch['within_1st']*100:6.2f}%  ⭐")

    # Onset metrics
    onset = results['onset_detailed']
    print("\n📊 ONSET DETECTION METRICS")
    print("-" * 80)
    print(f"  Total predicted events:      {onset['pred_event_count']:,}")
    print(f"  Total true events:           {onset['true_event_count']:,}")
    print(f"  Detection rate:              {onset['pred_event_count']/max(onset['true_event_count'],1)*100:.1f}%")
    print()
    print(f"  Frame-level (exact, ~32ms):")
    print(f"    Precision:                 {onset['frame_level']['precision']*100:6.2f}%")
    print(f"    Recall:                    {onset['frame_level']['recall']*100:6.2f}%")
    print(f"    F1:                        {onset['frame_level']['f1']*100:6.2f}%")
    print()
    print(f"  Event-based (±25ms tolerance):")
    print(f"    Precision:                 {onset['event_25ms']['precision']*100:6.2f}%")
    print(f"    Recall:                    {onset['event_25ms']['recall']*100:6.2f}%")
    print(f"    F1:                        {onset['event_25ms']['f1']*100:6.2f}%  ⭐")
    print(f"    Mean timing error:         {onset['event_25ms']['mean_timing_error_ms']:.1f} ms")
    print()
    print(f"  Event-based (±50ms tolerance):")
    print(f"    Precision:                 {onset['event_50ms']['precision']*100:6.2f}%")
    print(f"    Recall:                    {onset['event_50ms']['recall']*100:6.2f}%")
    print(f"    F1:                        {onset['event_50ms']['f1']*100:6.2f}%  ⭐")
    print(f"    Mean timing error:         {onset['event_50ms']['mean_timing_error_ms']:.1f} ms")
    print()
    print(f"  Event-based (±100ms tolerance):")
    print(f"    Precision:                 {onset['event_100ms']['precision']*100:6.2f}%")
    print(f"    Recall:                    {onset['event_100ms']['recall']*100:6.2f}%")
    print(f"    F1:                        {onset['event_100ms']['f1']*100:6.2f}%  ⭐")
    print(f"    Mean timing error:         {onset['event_100ms']['mean_timing_error_ms']:.1f} ms")

    # Offset metrics
    offset = results['offset_detailed']
    print("\n📊 OFFSET DETECTION METRICS")
    print("-" * 80)
    print(f"  Total predicted events:      {offset['pred_event_count']:,}")
    print(f"  Total true events:           {offset['true_event_count']:,}")
    print(f"  Detection rate:              {offset['pred_event_count']/max(offset['true_event_count'],1)*100:.1f}%")
    print()
    print(f"  Frame-level (exact, ~32ms):")
    print(f"    Precision:                 {offset['frame_level']['precision']*100:6.2f}%")
    print(f"    Recall:                    {offset['frame_level']['recall']*100:6.2f}%")
    print(f"    F1:                        {offset['frame_level']['f1']*100:6.2f}%")
    print()
    print(f"  Event-based (±25ms tolerance):")
    print(f"    Precision:                 {offset['event_25ms']['precision']*100:6.2f}%")
    print(f"    Recall:                    {offset['event_25ms']['recall']*100:6.2f}%")
    print(f"    F1:                        {offset['event_25ms']['f1']*100:6.2f}%  ⭐")
    print(f"    Mean timing error:         {offset['event_25ms']['mean_timing_error_ms']:.1f} ms")
    print()
    print(f"  Event-based (±50ms tolerance):")
    print(f"    Precision:                 {offset['event_50ms']['precision']*100:6.2f}%")
    print(f"    Recall:                    {offset['event_50ms']['recall']*100:6.2f}%")
    print(f"    F1:                        {offset['event_50ms']['f1']*100:6.2f}%  ⭐")
    print(f"    Mean timing error:         {offset['event_50ms']['mean_timing_error_ms']:.1f} ms")
    print()
    print(f"  Event-based (±100ms tolerance):")
    print(f"    Precision:                 {offset['event_100ms']['precision']*100:6.2f}%")
    print(f"    Recall:                    {offset['event_100ms']['recall']*100:6.2f}%")
    print(f"    F1:                        {offset['event_100ms']['f1']*100:6.2f}%  ⭐")
    print(f"    Mean timing error:         {offset['event_100ms']['mean_timing_error_ms']:.1f} ms")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    onset_50 = onset['event_50ms']['f1'] * 100
    offset_50 = offset['event_50ms']['f1'] * 100
    print(f"✅ Pitch detection: {pitch['within_1st']*100:.1f}% (excellent!)")
    print(f"✅ Onset detection (±50ms): {onset_50:.1f}% F1")
    print(f"✅ Offset detection (±50ms): {offset_50:.1f}% F1")
    if onset_50 >= 50:
        print(f"\n⭐ With ±50ms tolerance, onset detection is GOOD for melody transcription!")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--manifest', type=str, default='dataset/combined_manifest.json')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output', type=str, default='combined_detailed_evaluation.json')
    parser.add_argument('--val-split', type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading combined model from: {args.checkpoint}")
    model = load_combined_model(args.checkpoint, device=str(device))

    # Load dataset
    print(f"\nLoading dataset from: {args.manifest}")
    full_dataset = EnhancedMelodyDataset(
        labels_path=args.manifest,
        n_bins=88,
        augment=False,
        spec_augment=False,
        use_onset_features=True,
        use_musical_context=True
    )

    # Split
    train_size = int((1.0 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Using validation set: {val_size} samples")

    # DataLoader
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Evaluate
    print("\nRunning detailed evaluation...")
    results = evaluate_model_detailed(model, dataloader, device)

    # Print results
    print_detailed_results(results)

    # Save
    # Remove timing_errors lists (too large for JSON)
    for key in ['onset_detailed', 'offset_detailed']:
        for metric_key in results[key]:
            if isinstance(results[key][metric_key], dict) and 'timing_errors' in results[key][metric_key]:
                del results[key][metric_key]['timing_errors']

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
