#!/usr/bin/env python3
"""Experiment 5A: evaluate a frozen imitation CNN at multiple input brightness levels.

Requires train_imitation_cnn.py in the same directory. No labels or training.
Brightness is applied ONLY to the RGB frame input, never to the Doom environment.
"""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import vizdoom as vzd
from train_imitation_cnn import DoomImitationCNN, ACTIONS, TIC_CLASSES, create_game, preprocess


def wilson(successes, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n))/denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return [round(c-half, 6), round(c+half, 6)]


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def evaluate_condition(model, device, multiplier, episodes, headless):
    game = create_game(not headless)
    rows = []
    try:
        for i in range(1, episodes + 1):
            game.new_episode()
            start_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            reward_total = 0.0
            counts = [0, 0, 0]
            durations = {v: 0 for v in TIC_CLASSES}
            decisions = 0
            with torch.inference_mode():
                while not game.is_episode_finished():
                    state = game.get_state()
                    if state is None:
                        break
                    frame = preprocess(state.screen_buffer)
                    # RGB is uint8 CHW; clip after multiplying to avoid overflow.
                    transformed = np.clip(frame.astype(np.float32) * multiplier, 0, 255) / 255.0
                    inp = torch.from_numpy(transformed).unsqueeze(0).to(device)
                    logits_action, logits_duration = model(inp)
                    action_idx = int(logits_action.argmax(dim=1).item())
                    duration = TIC_CLASSES[int(logits_duration.argmax(dim=1).item())]
                    reward_total += float(game.make_action(list(ACTIONS[action_idx]), duration))
                    counts[action_idx] += 1
                    durations[duration] += 1
                    decisions += 1
            kills = max(0, int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)) - start_kills)
            rows.append({
                'brightness': multiplier, 'episode': i, 'reward': round(reward_total, 3),
                'kills': kills, 'success': int(kills > 0), 'decisions': decisions,
                'move_left': counts[0], 'move_right': counts[1], 'attack': counts[2],
                **{f'tics_{v}': durations[v] for v in TIC_CLASSES},
            })
            if i % 25 == 0 or i == episodes:
                s = sum(r['success'] for r in rows)
                print(f'  brightness {multiplier:.2f} | {i}/{episodes} | kills {s}/{i}', flush=True)
    finally:
        game.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/imitation_cnn/best.pt'))
    parser.add_argument('--out-dir', type=Path, default=Path('runs/brightness_robustness'))
    parser.add_argument('--episodes', type=int, default=100, help='Episodes PER brightness setting')
    parser.add_argument('--brightness', nargs='+', type=float, default=[1.0, 0.5, 0.75, 1.25, 1.5])
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.episodes < 1 or args.threads < 1 or any(x <= 0 or not math.isfinite(x) for x in args.brightness):
        parser.error('Episodes/threads must be positive and brightness values must be finite and >0')
    if len(args.brightness) != len(set(args.brightness)):
        parser.error('Brightness values must be unique')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if not Path('train_imitation_cnn.py').exists():
        parser.error('Run from CNN-build with train_imitation_cnn.py in this directory')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.out_dir / 'episodes.csv'
    summary_path = args.out_dir / 'summary.csv'
    metadata_path = args.out_dir / 'metadata.json'
    if any(p.exists() for p in (episode_path, summary_path, metadata_path)):
        parser.error(f'Output exists in {args.out_dir}. Choose a fresh --out-dir to protect results.')

    torch.set_num_threads(args.threads)
    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DoomImitationCNN().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Checkpoint {args.checkpoint} (epoch {ckpt.get("epoch", "unknown")}); device={device}')
    print('Frozen model; no labels; brightness applied to CNN input only.')

    all_rows = []
    summaries = []
    for mult in args.brightness:
        print(f'\nTesting brightness {mult:.2f}x')
        rows = evaluate_condition(model, device, mult, args.episodes, args.headless)
        all_rows.extend(rows)
        n = len(rows)
        successes = sum(r['success'] for r in rows)
        rewards = np.asarray([r['reward'] for r in rows], dtype=float)
        decisions = np.asarray([r['decisions'] for r in rows], dtype=float)
        summaries.append({
            'brightness': mult, 'episodes': n, 'kills': successes,
            'success_rate': round(successes / n, 6),
            'wilson_low': wilson(successes, n)[0], 'wilson_high': wilson(successes, n)[1],
            'mean_reward': round(float(rewards.mean()), 3),
            'median_reward': round(float(np.median(rewards)), 3),
            'min_reward': round(float(rewards.min()), 3),
            'mean_decisions': round(float(decisions.mean()), 3),
            'move_left': sum(r['move_left'] for r in rows),
            'move_right': sum(r['move_right'] for r in rows),
            'attack': sum(r['attack'] for r in rows),
        })

    for path, rows in ((episode_path, all_rows), (summary_path, summaries)):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checksum(args.checkpoint),
        'checkpoint_epoch': ckpt.get('epoch'), 'device': str(device),
        'episodes_per_condition': args.episodes, 'brightness_multipliers': args.brightness,
        'brightness_applied_to': 'preprocessed RGB frames before CNN inference',
        'label_buffer_used': False, 'gradient_updates': 0,
        'paired_start_states': False,
        'note': 'Each condition uses new randomly generated episodes; comparisons are unpaired.',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print('\nBRIGHTNESS RESULTS')
    for r in summaries:
        print(f"{r['brightness']:.2f}x: kills {r['kills']}/{r['episodes']} ({100*r['success_rate']:.1f}%), "
              f"mean reward {r['mean_reward']:.2f}, decisions {r['mean_decisions']:.2f}")
    print(f'\nSaved: {episode_path}, {summary_path}, {metadata_path}')


if __name__ == '__main__':
    main()
