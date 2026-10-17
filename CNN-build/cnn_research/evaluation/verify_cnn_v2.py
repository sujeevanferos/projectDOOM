#!/usr/bin/env python3
"""Experiment 5C follow-up: frozen CNN v2 robustness verification.

Runs 1000 normal-brightness episodes and 1000 episodes at each of 0.5x and
0.75x brightness by default. All evaluations are CNN-only (no labels, no
random actions, no optimizer or gradient updates).

Run from CNN-build (with train_imitation_cnn.py alongside this file):
    python verify_cnn_v2.py --headless
    python verify_cnn_v2.py --headless --normal-episodes 1000 --dark-episodes 500

Outputs in runs/cnn_v2_verification/ by default; refuses to overwrite data.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from cnn_research.training.train_imitation_cnn import DoomImitationCNN
from cnn_research.evaluation.evaluate_brightness_robustness import evaluate_condition, wilson, checksum


def summarize(rows, brightness):
    if not rows:
        raise ValueError('No episodes recorded')
    n = len(rows)
    wins = sum(int(row['success']) for row in rows)
    rewards = np.asarray([row['reward'] for row in rows], dtype=float)
    decisions = np.asarray([row['decisions'] for row in rows], dtype=float)
    low, high = wilson(wins, n)
    return {
        'brightness': brightness,
        'episodes': n,
        'kills': wins,
        'success_rate': round(wins / n, 6),
        'wilson_low': low,
        'wilson_high': high,
        'mean_reward': round(float(np.mean(rewards)), 3),
        'median_reward': round(float(np.median(rewards)), 3),
        'min_reward': round(float(np.min(rewards)), 3),
        'max_reward': round(float(np.max(rewards)), 3),
        'mean_decisions': round(float(np.mean(decisions)), 3),
        'median_decisions': round(float(np.median(decisions)), 3),
        'move_left': sum(int(r['move_left']) for r in rows),
        'move_right': sum(int(r['move_right']) for r in rows),
        'attack': sum(int(r['attack']) for r in rows),
        'tics_2': sum(int(r['tics_2']) for r in rows),
        'tics_4': sum(int(r['tics_4']) for r in rows),
        'tics_8': sum(int(r['tics_8']) for r in rows),
    }


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('runs/imitation_cnn_brightness/best.pt'))
    parser.add_argument('--out-dir', type=Path,
                        default=Path('runs/cnn_v2_verification'))
    parser.add_argument('--normal-episodes', type=int, default=1000)
    parser.add_argument('--dark-episodes', type=int, default=1000,
                        help='Episodes EACH at 0.50x and 0.75x brightness')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()

    if min(args.normal_episodes, args.dark_episodes, args.threads) < 1:
        parser.error('Episode counts and thread count must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Cannot find checkpoint: {args.checkpoint}')
    output_files = [args.out_dir / name for name in ('episodes.csv', 'summary.csv', 'metadata.json')]
    if any(path.exists() for path in output_files):
        parser.error(f'Existing verification output in {args.out_dir}; choose a fresh --out-dir')

    torch.set_num_threads(args.threads)
    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DoomImitationCNN().to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    conditions = [(1.0, args.normal_episodes), (0.5, args.dark_episodes), (0.75, args.dark_episodes)]
    print(f'CNN v2 checkpoint: {args.checkpoint} (epoch {ckpt.get("epoch", "unknown")})')
    print(f'Device: {device}; weights frozen; labels disabled; no training')
    print(f'Total evaluation episodes: {sum(n for _, n in conditions)}', flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episode_file, summary_file, metadata_file = output_files
    summaries = []
    # Write completed condition data immediately, so partial results survive interruption.
    with episode_file.open('w', newline='') as out:
        writer = None
        for brightness, episodes in conditions:
            print(f'\nEvaluating brightness {brightness:.2f}x ({episodes} episodes)', flush=True)
            rows = evaluate_condition(model, device, brightness, episodes, args.headless)
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            writer.writerows(rows)
            out.flush()
            summary = summarize(rows, brightness)
            summaries.append(summary)
            # Save checkpointed summaries after each condition.
            write_csv(summary_file, summaries)
            print(f'Finished {brightness:.2f}x: {summary["kills"]}/{episodes} kills '
                  f'({100*summary["success_rate"]:.2f}%), '
                  f'mean reward={summary["mean_reward"]:.2f}, '
                  f'mean decisions={summary["mean_decisions"]:.2f}', flush=True)

    metadata = {
        'experiment': 'CNN v2 post-augmentation robustness verification',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': checksum(args.checkpoint),
        'checkpoint_epoch': ckpt.get('epoch'),
        'device': str(device),
        'conditions': [{'brightness': b, 'episodes': n} for b, n in conditions],
        'brightness_applied_to': 'CNN RGB input only; Doom game appearance unchanged',
        'cnn_only': True, 'labels_used': False, 'gradient_updates': 0,
        'paired_start_states': False,
        'note': 'Separate random game episodes by condition. Comparisons to prior CNN v1 runs are unpaired.',
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + '\n')
    print('\nVERIFICATION SUMMARY')
    for s in summaries:
        print(f'{s["brightness"]:.2f}x: {s["kills"]}/{s["episodes"]} '
              f'({s["success_rate"]*100:.2f}%), 95% Wilson '
              f'[{s["wilson_low"]*100:.2f}%, {s["wilson_high"]*100:.2f}%], '
              f'reward={s["mean_reward"]:.2f}, decisions={s["mean_decisions"]:.2f}')
    print(f'\nSaved to: {args.out_dir}')


if __name__ == '__main__':
    main()
