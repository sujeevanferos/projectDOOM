#!/usr/bin/env python3
"""Minimal, trainable CNN + vanilla DQN baseline for ViZDoom basic.wad.

Train: python doom_dqn_baseline.py --episodes 1000 --headless
Watch: python doom_dqn_baseline.py --eval --checkpoint runs/basic_dqn/best.pt
Resume: python doom_dqn_baseline.py --episodes 500 --resume runs/basic_dqn/latest.pt --headless

Requires: pip install vizdoom torch numpy
"""
import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import vizdoom as vzd

# Button order matches basic.py: MOVE_LEFT, MOVE_RIGHT, ATTACK.
ACTIONS = ([True, False, False], [False, True, False], [False, False, True])


class DoomCNN(nn.Module):
    """Three convolution/pooling blocks followed by an action-value head."""
    def __init__(self, num_actions=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 4, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def forward(self, frames):
        return self.head(self.features(frames))


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # Store uint8 images; convert to float only when sampled.
        self.data.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size, device):
        samples = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.from_numpy(np.stack(states)).to(device=device, dtype=torch.float32) / 255.0
        next_states = torch.from_numpy(np.stack(next_states)).to(device=device, dtype=torch.float32) / 255.0
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


def preprocess(screen):
    """RGB24 -> uint8 CHW, 60x80 via 2x spatial subsampling.

    ViZDoom still renders at requested RES_160X120. Reduced CNN input and
    replay-memory footprint make this baseline easier to run on a laptop.
    """
    img = np.asarray(screen)
    if img.shape == (3, 120, 160):
        return np.ascontiguousarray(img[:, ::2, ::2])
    if img.shape == (120, 160, 3):
        return np.ascontiguousarray(img[::2, ::2, :].transpose(2, 0, 1))
    raise ValueError(f"Unexpected ViZDoom frame shape {img.shape}; expected RGB24 at 160x120")


def create_game(visible):
    game = vzd.DoomGame()
    game.set_doom_scenario_path(str(Path(vzd.scenarios_path) / 'basic.wad'))
    game.set_doom_map('map01')
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_living_reward(-1)
    game.set_render_hud(False)
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_labels_buffer_enabled(True)
    game.init()
    return game


def choose_action(model, state, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    with torch.no_grad():
        x = torch.from_numpy(state.copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
        return int(model(x).argmax(dim=1).item())


def train_step(online, target, replay, optimizer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = replay.sample(batch_size, device)
    q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Vanilla DQN: target network supplies the highest next-state value.
        max_next_q = target(next_states).max(dim=1).values
        target_q = rewards + gamma * (1.0 - dones) * max_next_q
    loss = F.smooth_l1_loss(q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def play_episode(game, online, device, frame_skip, epsilon=0.0, replay=None,
                 train_callback=None):
    game.new_episode()

    doom_state = game.get_state()

    if doom_state is not None:
        for label in doom_state.labels or []:
            print(
                "Object:", label.object_name,
                "Category:", label.object_category,
                "X:", label.x,
                "Y:", label.y,
                "Width:", label.width,
                "Height:", label.height
            )

    state = preprocess(game.get_state().screen_buffer)
    total_reward = 0.0
    actions_taken = [0] * len(ACTIONS)
    losses = []
    decisions = 0
    while not game.is_episode_finished():
        action_idx = choose_action(online, state, epsilon, device)
        reward = float(game.make_action(ACTIONS[action_idx], frame_skip))
        done = game.is_episode_finished()
        next_state = np.zeros_like(state) if done else preprocess(game.get_state().screen_buffer)
        if replay is not None:
            replay.add(state, action_idx, reward, next_state, done)
            if train_callback is not None:
                loss = train_callback()
                if loss is not None:
                    losses.append(loss)
        total_reward += reward
        actions_taken[action_idx] += 1
        decisions += 1
        state = next_state
    return dict(reward=total_reward, decisions=decisions,
                loss=float(np.mean(losses)) if losses else float('nan'),
                actions=actions_taken)


def save_checkpoint(path, model, target, optimizer, episode, steps, best_eval, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(), 'target': target.state_dict(),
        'optimizer': optimizer.state_dict(), 'episode': episode,
        'steps': steps, 'best_eval': best_eval,
        'architecture': 'DoomCNN_RGB_3actions', 'frame_skip': args.frame_skip,
    }, path)


def load_checkpoint(path, model, device, target=None, optimizer=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if target is not None:
        target.load_state_dict(checkpoint.get('target', checkpoint['model']))
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.set_num_threads(max(1, args.threads))
    print(f'Device: {device} | ViZDoom: RES_160X120 | CNN input: RGB 3x60x80')
    model = DoomCNN().to(device)
    target = DoomCNN().to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    game = create_game(visible=not args.headless)
    try:
        if args.eval:
            ckpt = load_checkpoint(args.checkpoint, model, device)
            print(f"Evaluating checkpoint from episode {ckpt.get('episode', '?')}")
            scores = []
            for idx in range(1, args.episodes + 1):
                stats = play_episode(game, model, device, args.frame_skip)
                scores.append(stats['reward'])
                print(f"Evaluation {idx}: reward={stats['reward']:.1f} "
                      f"decisions={stats['decisions']} actions={stats['actions']}")
            print(f'Mean evaluation reward: {np.mean(scores):.2f}')
            return

        args.out_dir.mkdir(parents=True, exist_ok=True)
        replay = ReplayBuffer(args.replay_size)
        start_episode = 0
        steps = 0
        best_eval = float('-inf')
        if args.resume:
            ckpt = load_checkpoint(args.resume, model, device, target, optimizer)
            start_episode = int(ckpt.get('episode', 0))
            steps = int(ckpt.get('steps', 0))
            best_eval = float(ckpt.get('best_eval', float('-inf')))
            print('Resuming weights, optimizer and epsilon schedule. '
                  'Replay buffer starts empty and refills.')
        csv_path = args.out_dir / 'history.csv'
        with csv_path.open('a', newline='') as fp:
            writer = csv.writer(fp)
            if fp.tell() == 0:
                writer.writerow(['episode', 'steps', 'epsilon', 'train_reward',
                                 'mean_td_loss', 'move_left', 'move_right',
                                 'attack', 'eval_mean_reward'])
            for episode in range(start_episode + 1, start_episode + args.episodes + 1):
                epsilon = max(args.epsilon_end,
                              args.epsilon_start - (args.epsilon_start - args.epsilon_end)
                              * steps / max(1, args.epsilon_decay_steps))

                def learn():
                    nonlocal steps
                    steps += 1
                    loss = None
                    if len(replay) >= max(args.batch_size, args.learning_starts) and steps % args.train_every == 0:
                        loss = train_step(model, target, replay, optimizer,
                                          args.batch_size, args.gamma, device)
                    if steps % args.target_update == 0:
                        target.load_state_dict(model.state_dict())
                    return loss

                stats = play_episode(game, model, device, args.frame_skip,
                                     epsilon, replay, learn)
                eval_score = ''
                if episode % args.eval_every == 0:
                    scores = [play_episode(game, model, device, args.frame_skip)['reward']
                              for _ in range(args.eval_episodes)]
                    eval_score = float(np.mean(scores))
                    if eval_score > best_eval:
                        best_eval = eval_score
                        save_checkpoint(args.out_dir / 'best.pt', model, target,
                                        optimizer, episode, steps, best_eval, args)
                if episode % args.save_every == 0 or episode == start_episode + args.episodes:
                    save_checkpoint(args.out_dir / 'latest.pt', model, target,
                                    optimizer, episode, steps, best_eval, args)
                writer.writerow([episode, steps, round(epsilon, 4), stats['reward'],
                                 stats['loss'], *stats['actions'], eval_score])
                fp.flush()
                print(f"Ep {episode:4d} | reward {stats['reward']:7.1f} | "
                      f"epsilon {epsilon:.3f} | loss {stats['loss']:.4f} | "
                      f"actions {stats['actions']}" +
                      (f' | eval {eval_score:.1f}' if eval_score != '' else ''))
        print(f'Done. Checkpoints and history: {args.out_dir}')
    finally:
        game.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--headless', action='store_true', help='Hide the game window')
    parser.add_argument('--eval', action='store_true', help='Only evaluate, no training/exploration')
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/basic_dqn/best.pt'))
    parser.add_argument('--resume', type=Path, help='Resume checkpoint (replay memory not persisted)')
    parser.add_argument('--out-dir', type=Path, default=Path('runs/basic_dqn'))
    parser.add_argument('--frame-skip', type=int, default=4, help='Game tics per selected action')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--replay-size', type=int, default=3000)
    parser.add_argument('--learning-starts', type=int, default=500)
    parser.add_argument('--train-every', type=int, default=4)
    parser.add_argument('--target-update', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--epsilon-decay-steps', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.frame_skip < 1 or args.episodes < 1 or args.eval_every < 1 or args.save_every < 1:
        parser.error('frame-skip, episodes, eval-every and save-every must be positive')
    return args


if __name__ == '__main__':
    run(parse_args())
