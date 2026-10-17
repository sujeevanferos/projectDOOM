#!/usr/bin/env python3
"""Experiment 5C: train CNN v2 with brightness-augmented RGB demonstrations.

Uses original DoomImitationCNN architecture, labels, split and hyperparameters.
Training samples ONLY receive random RGB brightness augmentation; validation
samples remain unmodified. The original v1 checkpoint is never touched.

Run:
 python -m cnn_research.training.train_imitation_cnn_brightness --epochs 50
Then evaluate with:
 python -m cnn_research.evaluation.evaluate_brightness_robustness --checkpoint runs/imitation_cnn_brightness/best.pt --out-dir runs/brightness_robustness_v2 --episodes 100 --headless
"""
import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cnn_research.training.train_imitation_cnn import (
    DoomImitationCNN, FrameDataset, load_demos, epoch_pass, TIC_CLASSES
)


class BrightnessTrainingDataset(Dataset):
    def __init__(self, base, low=0.5, high=1.5, probability=0.8):
        self.base = base
        self.low = low
        self.high = high
        self.probability = probability

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, action, duration = self.base[index]
        if random.random() < self.probability:
            multiplier = random.uniform(self.low, self.high)
            image = (image * multiplier).clamp(0.0, 1.0)
        return image, action, duration


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--demo-dir', type=Path, default=Path('runs/labelled_teacher/demos'))
    p.add_argument('--out-dir', type=Path, default=Path('runs/imitation_cnn_brightness'))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--duration-weight', type=float, default=0.4)
    p.add_argument('--brightness-min', type=float, default=0.5)
    p.add_argument('--brightness-max', type=float, default=1.5)
    p.add_argument('--augment-prob', type=float, default=0.8,
                   help='Probability a training frame receives random brightness; remaining frames stay original')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    if not (args.epochs >= 1 and args.batch_size >= 1 and args.patience >= 1 and args.threads >= 1):
        p.error('epochs, batch-size, patience, and threads must be positive')
    if not (0 < args.val_fraction < 1 and 0 <= args.augment_prob <= 1 and args.duration_weight >= 0):
        p.error('val-fraction must be (0,1), augment-prob [0,1], duration-weight >=0')
    if not (0 < args.brightness_min <= args.brightness_max and all(map(math.isfinite, [args.brightness_min, args.brightness_max, args.lr])) and args.lr > 0):
        p.error('brightness bounds must be finite and positive, and lr > 0')
    if args.out_dir.resolve() == Path('runs/imitation_cnn').resolve():
        p.error('Refusing to overwrite the existing CNN v1 directory')
    outputs = [args.out_dir / x for x in ('best.pt', 'history.csv', 'metadata.json', 'split_manifest.json')]
    if any(x.exists() for x in outputs):
        p.error(f'Existing output files in {args.out_dir}; choose a fresh --out-dir')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; brightness range {args.brightness_min}–{args.brightness_max}; probability {args.augment_prob}')

    paths = sorted(args.demo_dir.glob('demo_*.npz'))
    episodes = load_demos(args.demo_dir)
    # Exact same episode split and seed as original trainer.
    indices = np.random.default_rng(args.seed).permutation(len(episodes))
    n_val = max(1, round(len(episodes) * args.val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_eps = [episodes[int(i)] for i in train_indices]
    val_eps = [episodes[int(i)] for i in val_indices]
    train_ds = BrightnessTrainingDataset(FrameDataset(train_eps), args.brightness_min,
                                         args.brightness_max, args.augment_prob)
    val_ds = FrameDataset(val_eps)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')
    print(f'Episode split: train={len(train_eps)} validation={len(val_eps)}')
    print(f'Decision split: train={len(train_ds)} validation={len(val_ds)}')
    model = DoomImitationCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'split_manifest.json').write_text(json.dumps({
        'train_files': [paths[int(i)].name for i in train_indices],
        'validation_files': [paths[int(i)].name for i in val_indices],
        'seed': args.seed, 'val_fraction': args.val_fraction
    }, indent=2))
    metadata = {
        'experiment': '5C brightness-augmented imitation CNN v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_script': 'train_imitation_cnn.py',
        'demo_dir': str(args.demo_dir), 'episodes': len(episodes),
        'train_episodes': len(train_eps), 'val_episodes': len(val_eps),
        'train_samples': len(train_ds), 'val_samples': len(val_ds),
        'augmentation': {'training_only': True, 'type': 'RGB multiplicative brightness',
                         'min': args.brightness_min, 'max': args.brightness_max,
                         'probability': args.augment_prob},
        'validation_augmentation': False,
        'seed': args.seed, 'device': str(device), 'epochs_max': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr, 'patience': args.patience,
        'duration_weight': args.duration_weight,
        'architecture': 'DoomImitationCNN_v1 (unchanged)',
        'teacher_labels_supplied_to_CNN': False,
        'note': 'Uses same split procedure as CNN v1 with same seed and demonstration directory.'
    }
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    best_loss, stale = float('inf'), 0
    best_path = args.out_dir / 'best.pt'
    with (args.out_dir / 'history.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_action_accuracy', 'train_duration_accuracy',
                         'val_loss', 'val_action_accuracy', 'val_duration_accuracy'])
        for epoch in range(1, args.epochs + 1):
            tr = epoch_pass(model, train_loader, device, optimizer, args.duration_weight)
            va = epoch_pass(model, val_loader, device, None, args.duration_weight)
            writer.writerow([epoch, *[round(x, 6) for x in (*tr, *va)]])
            f.flush()
            print(f'Epoch {epoch:3d}/{args.epochs} | train loss={tr[0]:.4f} act={tr[1]:.1%} tic={tr[2]:.1%} | '
                  f'val loss={va[0]:.4f} act={va[1]:.1%} tic={va[2]:.1%}', flush=True)
            if va[0] < best_loss - 1e-5:
                best_loss, stale = va[0], 0
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': best_loss,
                            'tic_classes': TIC_CLASSES, 'seed': args.seed,
                            'architecture': 'DoomImitationCNN_v1',
                            'augmentation': metadata['augmentation']}, best_path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
    metadata['checkpoint_sha256'] = sha256_file(best_path)
    metadata['best_val_loss'] = best_loss
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Checkpoint: {best_path}\nHistory: {args.out_dir / "history.csv"}')
    print('Evaluate with python -m cnn_research.evaluation.evaluate_brightness_robustness using a new --out-dir')


if __name__ == '__main__':
    main()
