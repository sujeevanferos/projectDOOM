#!/usr/bin/env python3
"""Train and evaluate a ViZDoom CNN from label-guided teacher demonstrations.

Examples:
  python -m cnn_research.training.train_imitation_cnn --epochs 50
  python -m cnn_research.training.train_imitation_cnn --eval --episodes 100 --headless
  python -m cnn_research.training.train_imitation_cnn --eval --episodes 5

Input NPZ fields: frames (N,3,60,80) uint8, actions (N,),
rewards (N,), action_tics (N,). Labels are NOT used by the CNN.

The CNN predicts both action (left/right/attack) and action duration,
so it can imitate the teacher's variable-length movement steps.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import vizdoom as vzd

ACTIONS = ((True, False, False), (False, True, False), (False, False, True))
ACTION_NAMES = ("LEFT", "RIGHT", "ATTACK")
TIC_CLASSES = (2, 4, 8)  # teacher's near / search-or-attack / far durations


class DoomImitationCNN(nn.Module):
    """Same convolutional backbone as the DQN baseline; two prediction heads."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
        )
        self.action_head = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 3))
        self.duration_head = nn.Sequential(nn.Linear(768, 64), nn.ReLU(), nn.Linear(64, len(TIC_CLASSES)))

    def forward(self, x):
        h = self.features(x)
        return self.action_head(h), self.duration_head(h)


class FrameDataset(Dataset):
    def __init__(self, episodes):
        self.frames = np.concatenate([e["frames"] for e in episodes], axis=0)
        self.actions = np.concatenate([e["actions"] for e in episodes], axis=0)
        duration_values = np.concatenate([e["action_tics"] for e in episodes], axis=0)
        lookup = {v: i for i, v in enumerate(TIC_CLASSES)}
        self.durations = np.array([lookup[int(v)] for v in duration_values], dtype=np.int64)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return (torch.from_numpy(self.frames[i].copy()).float().div_(255),
                int(self.actions[i]), int(self.durations[i]))


def load_demos(directory):
    paths = sorted(directory.glob("demo_*.npz"))
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 demonstration files in {directory}, found {len(paths)}")
    episodes = []
    action_totals = np.zeros(3, dtype=np.int64)
    tic_totals = np.zeros(len(TIC_CLASSES), dtype=np.int64)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"frames", "actions", "rewards", "action_tics"}
            if not required.issubset(data.files):
                raise ValueError(f"Missing fields in {path}: {required - set(data.files)}")
            frames = np.asarray(data["frames"], dtype=np.uint8)
            actions = np.asarray(data["actions"], dtype=np.int64)
            tics = np.asarray(data["action_tics"], dtype=np.int64)
            rewards = data["rewards"]
        n = len(actions)
        if n < 1 or frames.shape != (n, 3, 60, 80) or len(tics) != n or len(rewards) != n:
            raise ValueError(f"Invalid shapes in {path}: frames={frames.shape}, actions={actions.shape}")
        if np.any((actions < 0) | (actions > 2)) or not set(tics.tolist()).issubset(set(TIC_CLASSES)):
            raise ValueError(f"Unexpected action or duration in {path}: durations={set(tics.tolist())}")
        episodes.append({"frames": frames, "actions": actions, "action_tics": tics})
        action_totals += np.bincount(actions, minlength=3)
        tic_totals += np.bincount(np.searchsorted(TIC_CLASSES, tics), minlength=len(TIC_CLASSES))
    print(f"Loaded {len(episodes)} episodes; {sum(map(lambda e: len(e['actions']), episodes))} decisions")
    print(f"Action counts (L/R/ATTACK): {action_totals.tolist()}")
    print(f"Duration counts ({TIC_CLASSES}): {tic_totals.tolist()}")
    return episodes


def epoch_pass(model, loader, device, optimizer=None, duration_weight=0.4):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    action_ok = tic_ok = n = 0
    for images, actions, durations in loader:
        images = images.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        durations = durations.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits, tics_logits = model(images)
            loss = F.cross_entropy(logits, actions) + duration_weight * F.cross_entropy(tics_logits, durations)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
        batch_size = len(actions)
        n += batch_size
        total_loss += float(loss.item()) * batch_size
        action_ok += int((logits.argmax(1) == actions).sum().item())
        tic_ok += int((tics_logits.argmax(1) == durations).sum().item())
    return total_loss / n, action_ok / n, tic_ok / n


def create_game(visible):
    game = vzd.DoomGame()
    game.set_doom_scenario_path(str(Path(vzd.scenarios_path) / "basic.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_living_reward(-1)
    game.set_render_hud(False)
    game.set_render_weapon(True)
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.PLAYER)
    # Deliberately NO label buffer: only the RGB pixels are supplied to the student.
    game.init()
    return game


def preprocess(screen):
    img = np.asarray(screen)
    if img.shape == (3, 120, 160):
        return np.ascontiguousarray(img[:, ::2, ::2])
    if img.shape == (120, 160, 3):
        return np.ascontiguousarray(img[::2, ::2, :].transpose(2, 0, 1))
    raise ValueError(f"Unexpected screen shape {img.shape}")


def evaluate(model, device, episodes, visible, csv_path=None):
    game = create_game(visible)
    rows = []
    model.eval()
    try:
        for episode in range(1, episodes + 1):
            game.new_episode()
            start_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            total_reward, decisions = 0.0, 0
            counts = [0, 0, 0]
            duration_counts = {v: 0 for v in TIC_CLASSES}
            with torch.inference_mode():
                while not game.is_episode_finished():
                    gs = game.get_state()
                    if gs is None:
                        break
                    frame = preprocess(gs.screen_buffer)
                    tensor = torch.from_numpy(frame.copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
                    action_logits, duration_logits = model(tensor)
                    action = int(action_logits.argmax(1).item())
                    tics = TIC_CLASSES[int(duration_logits.argmax(1).item())]
                    # No exploration and no privileged enemy labels.
                    reward = float(game.make_action(list(ACTIONS[action]), tics))
                    total_reward += reward
                    counts[action] += 1
                    duration_counts[tics] += 1
                    decisions += 1
            kills = max(0, int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)) - start_kills)
            row = {"episode": episode, "reward": round(total_reward, 3), "kills": kills,
                   "success": int(kills > 0), "decisions": decisions,
                   "move_left": counts[0], "move_right": counts[1], "attack": counts[2],
                   **{f"tics_{v}": duration_counts[v] for v in TIC_CLASSES}}
            rows.append(row)
            print(f"Eval {episode:4d}/{episodes} | reward={total_reward:7.1f} "
                  f"| kills={kills} | decisions={decisions:3d} | actions={counts}", flush=True)
    finally:
        game.close()
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Evaluation results: {csv_path}")
    print(f"CNN-only success: {sum(r['success'] for r in rows)}/{len(rows)} "
          f"({100 * np.mean([r['success'] for r in rows]):.1f}%)")
    print(f"CNN-only mean reward: {np.mean([r['reward'] for r in rows]):.2f}")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo-dir", type=Path, default=Path("runs/labelled_teacher/demos"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/imitation_cnn"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--duration-weight", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval", action="store_true", help="Evaluate trained CNN only; no training")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=100, help="Number of independent evaluation episodes")
    p.add_argument("--headless", action="store_true", help="Hide ViZDoom window (recommended on servers)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or not 0 < args.val_fraction < 1:
        p.error("Require positive epochs/batch-size/patience and 0 < val-fraction < 1")
    if args.episodes < 1 or args.duration_weight < 0:
        p.error("Require episodes >= 1 and duration-weight >= 0")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; RGB input 3x60x80; durations={TIC_CLASSES}", flush=True)
    model = DoomImitationCNN().to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint or args.out_dir / "best.pt"
    if args.eval:
        if not best_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {best_path}")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded epoch {checkpoint.get('epoch', '?')} from {best_path}")
        evaluate(model, device, args.episodes, not args.headless, args.out_dir / "evaluation.csv")
        return

    if (args.out_dir / "history.csv").exists() or best_path.exists():
        raise FileExistsError(f"Training outputs already exist in {args.out_dir}. Use a fresh --out-dir to avoid overwriting.")
    episodes = load_demos(args.demo_dir)
    indices = np.random.default_rng(args.seed).permutation(len(episodes))
    n_val = max(1, round(len(episodes) * args.val_fraction))
    val_eps = [episodes[int(i)] for i in indices[:n_val]]
    train_eps = [episodes[int(i)] for i in indices[n_val:]]
    train_data, val_data = FrameDataset(train_eps), FrameDataset(val_eps)
    print(f"Episode split: train={len(train_eps)}, validation={len(val_eps)}")
    print(f"Decision split: train={len(train_data)}, validation={len(val_data)}")
    if len(train_data) < 2:
        raise ValueError("Training split is too small")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history_path = args.out_dir / "history.csv"
    best_loss, stale = float("inf"), 0
    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_action_accuracy", "train_duration_accuracy",
                         "val_loss", "val_action_accuracy", "val_duration_accuracy"])
        for epoch in range(1, args.epochs + 1):
            train_metrics = epoch_pass(model, train_loader, device, optimizer, args.duration_weight)
            val_metrics = epoch_pass(model, val_loader, device, None, args.duration_weight)
            writer.writerow([epoch, *[round(v, 6) for v in (*train_metrics, *val_metrics)]])
            f.flush()
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train loss={train_metrics[0]:.4f} act={train_metrics[1]:.1%} tic={train_metrics[2]:.1%} | "
                  f"val loss={val_metrics[0]:.4f} act={val_metrics[1]:.1%} tic={val_metrics[2]:.1%}", flush=True)
            if val_metrics[0] < best_loss - 1e-5:
                best_loss, stale = val_metrics[0], 0
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_loss": best_loss, "tic_classes": TIC_CLASSES,
                            "seed": args.seed, "architecture": "DoomImitationCNN_v1"}, best_path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping: no val-loss improvement for {stale} epochs")
                    break
    print(f"Saved best imitation checkpoint: {best_path}")
    print(f"Training history: {history_path}")
    print("Next: python -m cnn_research.training.train_imitation_cnn --eval --episodes 100 --headless")


if __name__ == "__main__":
    main()
