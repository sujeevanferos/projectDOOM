#!/usr/bin/env python3
"""Train and evaluate a CNN-based DQN agent on ViZDoom's basic scenario."""
import argparse
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import vizdoom as vzd

Transition = namedtuple('Transition', 'state action reward next_state done')
ACTIONS = ([True, False, False], [False, True, False], [False, False, True])


class DoomCNN(nn.Module):
    """Q network: one Q value per discrete action."""
    def __init__(self, num_actions=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 3 * 4, 128), nn.ReLU(), nn.Linear(128, num_actions))

    def forward(self, x):
        return self.head(self.features(x))


def preprocess(frame):
    """ViZDoom RGB24 HWC uint8 -> CHW uint8, stored compactly in replay."""
    frame = np.asarray(frame)
    if frame.shape == (120, 160, 3):
        return np.ascontiguousarray(frame.transpose(2, 0, 1))
    if frame.shape == (3, 120, 160):
        return np.ascontiguousarray(frame)
    raise ValueError(f'Unexpected frame shape: {frame.shape}')


def tensorize(frames, device):
    return torch.from_numpy(np.stack(frames)).to(device=device, dtype=torch.float32).div_(255.0)


def create_game(visible):
    game = vzd.DoomGame()
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, 'basic.wad'))
    game.set_doom_map('map01')
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_living_reward(-1)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game


def optimize(policy, target, optimizer, replay, batch_size, gamma, device):
    if len(replay) < batch_size:
        return None
    batch = random.sample(replay, batch_size)
    states = tensorize([t.state for t in batch], device)
    actions = torch.tensor([t.action for t in batch], device=device).long().unsqueeze(1)
    rewards = torch.tensor([t.reward for t in batch], device=device, dtype=torch.float32)
    dones = torch.tensor([t.done for t in batch], device=device, dtype=torch.bool)
    next_states = tensorize([t.next_state for t in batch], device)

    predicted_q = policy(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        # Double DQN: policy selects the next action; target evaluates it.
        next_actions = policy(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions).squeeze(1)
        expected_q = rewards + gamma * next_q * (~dones).float()

    loss = F.smooth_l1_loss(predicted_q, expected_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    policy = DoomCNN().to(device)
    target = DoomCNN().to(device)
    if args.eval:
        policy.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    target.load_state_dict(policy.state_dict())
    target.eval()
    policy.eval() if args.eval else policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    replay = deque(maxlen=args.replay_size)
    steps = 0
    game = create_game(visible=not args.headless)
    print(f'Device: {device} | Mode: {"evaluation" if args.eval else "training"}')
    try:
        for episode in range(1, args.episodes + 1):
            game.new_episode()
            state = preprocess(game.get_state().screen_buffer)
            losses = []
            while not game.is_episode_finished():
                epsilon = 0.0 if args.eval else args.eps_end + (args.eps_start - args.eps_end) * max(0.0, 1.0 - steps / args.eps_decay_steps)
                if not args.eval and random.random() < epsilon:
                    action_idx = random.randrange(len(ACTIONS))  # exploration during training only
                else:
                    with torch.no_grad():
                        action_idx = policy(tensorize([state], device)).argmax(dim=1).item()
                reward = game.make_action(ACTIONS[action_idx], args.frame_skip)
                done = game.is_episode_finished()
                next_state = preprocess(game.get_state().screen_buffer) if not done else np.zeros_like(state)
                if not args.eval:
                    replay.append(Transition(state, action_idx, reward, next_state, done))
                    if steps % args.train_every == 0:
                        loss = optimize(policy, target, optimizer, replay, args.batch_size, args.gamma, device)
                        if loss is not None:
                            losses.append(loss)
                    if steps % args.target_update == 0:
                        target.load_state_dict(policy.state_dict())
                state = next_state
                steps += 1
            avg_loss = float(np.mean(losses)) if losses else float('nan')
            print(f'Episode {episode:4d}/{args.episodes} | reward={game.get_total_reward():8.1f} | epsilon={epsilon:.3f} | loss={avg_loss:.4f}')
            if not args.eval and episode % args.save_every == 0:
                torch.save(policy.state_dict(), args.checkpoint)
        if not args.eval:
            torch.save(policy.state_dict(), args.checkpoint)
            print(f'Saved CNN weights to {args.checkpoint}')
    finally:
        game.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViZDoom basic scenario CNN DQN')
    parser.add_argument('--eval', action='store_true', help='Run trained CNN without random exploration')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--checkpoint', default='doom_cnn.pt')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--replay-size', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--train-every', type=int, default=4)
    parser.add_argument('--target-update', type=int, default=500)
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--eps-end', type=float, default=0.05)
    parser.add_argument('--eps-decay-steps', type=int, default=30000)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
