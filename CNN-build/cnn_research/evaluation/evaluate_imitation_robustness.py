#!/usr/bin/env python3
"""Fresh CNN-only ViZDoom robustness evaluation; never modifies trained weights.

Run from directory containing train_imitation_cnn.py:
  python evaluate_imitation_robustness.py --episodes 1000 --headless
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from cnn_research.training.train_imitation_cnn import DoomImitationCNN, evaluate


def wilson_interval(successes, n, z=1.95996398454):
    if n == 0:
        return [None, None]
    p = successes / n
    denom = 1 + z*z/n
    mid = (p + z*z/(2*n)) / denom
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [round(mid-width, 6), round(mid+width, 6)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/imitation_cnn/best.pt'))
    parser.add_argument('--out-dir', type=Path, default=Path('runs/imitation_robustness_1000'))
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be >=1')
    if not args.checkpoint.is_file():
        parser.error(f'checkpoint not found: {args.checkpoint}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / 'evaluation.csv'
    if path.exists():
        parser.error(f'{path} already exists. Choose a new --out-dir to protect earlier results.')
    torch.set_num_threads(max(1, args.threads))
    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DoomImitationCNN().to(device)
    model.load_state_dict(checkpoint['model'])
    print(f'Checkpoint: {args.checkpoint} (epoch {checkpoint.get("epoch", "unknown")})')
    print(f'Device: {device}; episodes: {args.episodes}; no teacher labels; no learning')
    rows = evaluate(model, device, args.episodes, not args.headless, path)
    n = len(rows)
    s = sum(int(r['success']) for r in rows)
    rewards = np.array([r['reward'] for r in rows], dtype=float)
    decisions = np.array([r['decisions'] for r in rows], dtype=float)
    action_counts = {k: sum(int(r[k]) for r in rows) for k in ('move_left','move_right','attack')}
    durations = {k: sum(int(r[k]) for r in rows) for k in ('tics_2','tics_4','tics_8')}
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint), 'checkpoint_epoch': checkpoint.get('epoch'),
        'device': str(device), 'episode_count': n, 'successes': s,
        'success_rate': s/n, 'success_rate_wilson_95': wilson_interval(s, n),
        'mean_reward': float(rewards.mean()), 'median_reward': float(np.median(rewards)),
        'min_reward': float(rewards.min()), 'max_reward': float(rewards.max()),
        'mean_decisions': float(decisions.mean()), 'median_decisions': float(np.median(decisions)),
        'action_counts': action_counts, 'duration_counts': durations,
        'label_buffer_used': False, 'gradient_updates': False,
    }
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== ROBUSTNESS SUMMARY ===')
    print(f"Kills: {s}/{n} ({100*s/n:.2f}%)")
    print(f"95% Wilson interval: {100*summary['success_rate_wilson_95'][0]:.2f}%–{100*summary['success_rate_wilson_95'][1]:.2f}%")
    print(f"Mean reward: {summary['mean_reward']:.2f}; mean decisions: {summary['mean_decisions']:.2f}")
    print(f'Actions: {action_counts}; durations: {durations}')
    print(f'Saved: {path} and {summary_path}')


if __name__ == '__main__':
    main()
