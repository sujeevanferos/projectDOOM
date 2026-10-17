#!/usr/bin/env python3
"""Compare CNN v1 and v2 brightness robustness summaries (unpaired episodes).

Usage:
 python compare_brightness_models.py \
   --v1 runs/brightness_robustness/summary.csv \
   --v2 runs/brightness_robustness_v2/summary.csv
"""
import argparse
import csv
from pathlib import Path


def read_summary(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    result = {}
    needed = {'brightness', 'episodes', 'success_rate', 'mean_reward', 'mean_decisions'}
    for row in rows:
        if not needed.issubset(row):
            raise ValueError(f'Missing columns in {path}: {needed - set(row)}')
        brightness = float(row['brightness'])
        if brightness in result:
            raise ValueError(f'Duplicate brightness {brightness} in {path}')
        result[brightness] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', type=Path, default=Path('runs/brightness_robustness/summary.csv'))
    parser.add_argument('--v2', type=Path, default=Path('runs/brightness_robustness_v2/summary.csv'))
    parser.add_argument('--out', type=Path, default=Path('runs/brightness_comparison_v1_vs_v2.csv'))
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'{args.out} exists. Choose --out to avoid overwriting.')
    v1, v2 = read_summary(args.v1), read_summary(args.v2)
    common = sorted(v1.keys() & v2.keys())
    if not common:
        parser.error('No shared brightness conditions')
    if set(v1) != set(v2):
        print('Warning: comparing only shared brightness conditions')
    records = []
    for brightness in common:
        a, b = v1[brightness], v2[brightness]
        s1, s2 = float(a['success_rate']), float(b['success_rate'])
        r1, r2 = float(a['mean_reward']), float(b['mean_reward'])
        d1, d2 = float(a['mean_decisions']), float(b['mean_decisions'])
        records.append({
            'brightness': brightness,
            'v1_episodes': int(a['episodes']), 'v2_episodes': int(b['episodes']),
            'v1_success_pct': round(100*s1, 2), 'v2_success_pct': round(100*s2, 2),
            'delta_success_percentage_points': round(100*(s2-s1), 2),
            'v1_mean_reward': r1, 'v2_mean_reward': r2, 'delta_reward': round(r2-r1, 3),
            'v1_mean_decisions': d1, 'v2_mean_decisions': d2,
            'delta_decisions': round(d2-d1, 3)
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print('Brightness | v1 success | v2 success | delta (pp) | delta reward | delta decisions')
    for r in records:
        print(f"{r['brightness']:>9.2f}x | {r['v1_success_pct']:>9.1f}% | {r['v2_success_pct']:>9.1f}% | "
              f"{r['delta_success_percentage_points']:>+9.1f} | {r['delta_reward']:>+12.2f} | {r['delta_decisions']:>+15.2f}")
    print('\nUnpaired independently sampled episode sets; differences are descriptive, not causal proof.')
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
