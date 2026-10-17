#!/usr/bin/env python3
"""Experiment 5B: visualize learned CNN features and action-specific Grad-CAM.

Requires train_imitation_cnn.py in the same folder. Uses the ORIGINAL trained
checkpoint; no training, labels, or weight updates are performed.

Example:
 python experiment_5b_feature_visualization.py --episodes 12 --samples 9 --headless
"""
import argparse
import csv
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from train_imitation_cnn import DoomImitationCNN, TIC_CLASSES, ACTIONS, ACTION_NAMES, create_game, preprocess

CONV_LAYERS = (("conv1", 0), ("conv2", 3), ("conv3", 6))


def choose_and_visualize(model, frame, device, factor, maps_count, destination, name):
    """Run inference and compute action Grad-CAM; returns numeric diagnostics."""
    original = frame.astype(np.float32)
    changed = np.clip(original * factor, 0, 255).astype(np.uint8)
    x = torch.from_numpy(changed.copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    activations = {}
    handles = []
    try:
        for layer_name, idx in CONV_LAYERS:
            def capture(_module, _inputs, output, key=layer_name):
                activations[key] = output
            handles.append(model.features[idx].register_forward_hook(capture))
        with torch.enable_grad():
            action_logits, duration_logits = model(x)
            action_idx = int(action_logits.argmax(dim=1).item())
            duration_idx = int(duration_logits.argmax(dim=1).item())
            # Conv3 features are still spatial; gradients identify regions driving
            # the SELECTED ACTION logit (not the duration decision).
            last_feature = activations['conv3']
            grad, = torch.autograd.grad(action_logits[0, action_idx], last_feature)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * last_feature).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            cam = cam.detach().cpu().numpy()
            cam = cam / (cam.max() + 1e-8)
            action_probs = torch.softmax(action_logits, 1)[0].detach().cpu().numpy()
            duration_probs = torch.softmax(duration_logits, 1)[0].detach().cpu().numpy()
            feature_arrays = {k: v[0].detach().cpu().numpy() for k, v in activations.items()}
    finally:
        for handle in handles:
            handle.remove()

    # Frame as H,W,C RGB; map indices align with the downsampled ViZDoom RGB input.
    rgb = np.transpose(changed, (1, 2, 0))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    axs[0].imshow(rgb)
    axs[0].set_title(f'RGB input ({factor:g}x)')
    axs[1].imshow(rgb)
    axs[1].imshow(cam, cmap='jet', alpha=0.48, vmin=0, vmax=1)
    axs[1].set_title('Action Grad-CAM overlay')
    axs[2].imshow(cam, cmap='magma', vmin=0, vmax=1)
    axs[2].set_title('Action Grad-CAM heatmap')
    for ax in axs: ax.axis('off')
    fig.suptitle(f'{name} | {ACTION_NAMES[action_idx]} ({action_probs[action_idx]:.1%}) | {TIC_CLASSES[duration_idx]} tics', fontsize=11)
    fig.savefig(destination / f'{name}_cam.png', dpi=155)
    plt.close(fig)

    fig, axs = plt.subplots(3, maps_count, figsize=(maps_count * 1.55, 5.2), constrained_layout=True)
    for row, (layer_name, _) in enumerate(CONV_LAYERS):
        features = feature_arrays[layer_name]
        for j in range(maps_count):
            ax = axs[row, j]
            fm = features[j]
            # Normalize each channel for visibility; absolute intensities are NOT comparable.
            denom = float(fm.max() - fm.min())
            norm = (fm - fm.min()) / (denom + 1e-8)
            ax.imshow(norm, cmap='viridis', vmin=0, vmax=1)
            ax.axis('off')
            if j == 0: ax.set_title(f'{layer_name}\nchannel {j}', fontsize=9)
            else: ax.set_title(f'channel {j}', fontsize=9)
    fig.suptitle(f'{name} | feature maps (each channel normalized separately)', fontsize=11)
    fig.savefig(destination / f'{name}_features.png', dpi=135)
    plt.close(fig)
    return dict(action=ACTION_NAMES[action_idx], action_idx=action_idx,
                duration=TIC_CLASSES[duration_idx],
                action_confidence=float(action_probs[action_idx]),
                duration_confidence=float(duration_probs[duration_idx]),
                prob_left=float(action_probs[0]), prob_right=float(action_probs[1]),
                prob_attack=float(action_probs[2]), cam_max=float(cam.max()),
                cam_mean=float(cam.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/imitation_cnn/best.pt'))
    parser.add_argument('--out-dir', type=Path, default=Path('runs/experiment_5b_features'))
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--samples', type=int, default=9, help='Maximum captured RGB frames')
    parser.add_argument('--brightness', type=float, nargs='+', default=[1.0, 0.5],
                        help='Run each captured frame at these brightness multipliers')
    parser.add_argument('--maps', type=int, default=8, help='Channels per conv layer, 1 to 16')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or args.samples < 1 or not 1 <= args.maps <= 16:
        parser.error('episodes/samples must be positive and maps must be 1..16')
    if any(not 0 < b <= 3 for b in args.brightness):
        parser.error('brightness factors must be >0 and <=3')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f'Output directory is not empty: {args.out_dir}; choose a new --out-dir')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model = DoomImitationCNN().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded epoch {ckpt.get("epoch", "?")}, device={device}; weights frozen')

    # Gather RGB frames from closed-loop CNN-only play, avoiding privileged labels.
    game = create_game(visible=not args.headless)
    frames = []
    total_kills = 0
    try:
        for ep in range(args.episodes):
            game.new_episode()
            start_kills = int(game.get_game_variable(__import__('vizdoom').GameVariable.KILLCOUNT))
            n = 0
            while not game.is_episode_finished():
                state = game.get_state()
                if state is None: break
                frame = preprocess(state.screen_buffer)
                # Spread examples over episodes where possible: capture initial and
                # at least one later frame, rather than only similar starting frames.
                if len(frames) < args.samples and (n == 0 or (n == 2 and len(frames) < args.samples)):
                    frames.append((ep + 1, n, frame.copy()))
                with torch.inference_mode():
                    x = torch.from_numpy(frame.copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
                    actions, durations = model(x)
                    a = int(actions.argmax(1).item())
                    t = TIC_CLASSES[int(durations.argmax(1).item())]
                game.make_action(list(ACTIONS[a]), t)
                n += 1
            total_kills += max(0, int(game.get_game_variable(__import__('vizdoom').GameVariable.KILLCOUNT)) - start_kills)
            if len(frames) >= args.samples: break
    finally:
        game.close()
    if not frames:
        raise RuntimeError('No frames were captured')

    records = []
    for i, (ep, step, frame) in enumerate(frames, 1):
        for factor in args.brightness:
            name = f'frame_{i:02d}_ep{ep:03d}_step{step:02d}_b{factor:g}'.replace('.', 'p')
            result = choose_and_visualize(model, frame, device, factor, args.maps, args.out_dir, name)
            records.append(dict(frame_id=i, episode=ep, step=step, brightness=factor, **result))
        print(f'Generated visualizations for captured frame {i}/{len(frames)}')
    with (args.out_dir / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    data = dict(experiment='5B CNN feature visualization', checkpoint=str(args.checkpoint),
                sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                checkpoint_epoch=ckpt.get('epoch'), device=str(device),
                episodes_started=ep, captured_frames=len(frames),
                games_killed_during_capture=total_kills, brightness_factors=args.brightness,
                maps_per_layer=args.maps, labels_used=False, gradient_updates=0,
                explanation='Grad-CAM is a qualitative attribution, not proof of object detection. Gradients are computed for analysis only; model weights are not updated.')
    (args.out_dir / 'metadata.json').write_text(json.dumps(data, indent=2) + '\n')
    print(f'Output: {args.out_dir}; visualizations={len(records)} x (CAM + feature maps)')
    print('Inspect *_cam.png, *_features.png, predictions.csv and metadata.json')


if __name__ == '__main__':
    main()
