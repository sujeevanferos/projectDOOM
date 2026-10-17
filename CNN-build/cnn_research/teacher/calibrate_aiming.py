#!/usr/bin/env python3
"""Calibrate the effect of ViZDoom MOVE_LEFT/MOVE_RIGHT on enemy screen position.

Run: python -m cnn_research.teacher.calibrate_aiming --episodes 5 --tics 2
This does NOT train a neural network.
"""
import argparse
from pathlib import Path
import vizdoom as vzd


def enemy_info(game):
    state = game.get_state()
    if state is None:
        return None
    monsters = [label for label in (state.labels or [])
                if label.object_category == 'Monster']
    if not monsters:
        return None
    enemy = max(monsters, key=lambda l: l.width * l.height)
    x = enemy.x + enemy.width / 2.0
    return x, x - 80.0, enemy.object_name


def fmt(info):
    if info is None:
        return 'enemy not visible'
    x, e, name = info
    return f'{name}: center_x={x:.1f}, error={e:+.1f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--tics', type=int, default=2, help='Game tics per calibration movement')
    args = parser.parse_args()
    if args.episodes < 1 or args.tics < 1:
        parser.error('--episodes and --tics must be positive')

    game = vzd.DoomGame()
    game.set_doom_scenario_path(str(Path(vzd.scenarios_path) / 'basic.wad'))
    game.set_doom_map('map01')
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_living_reward(-1)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_render_hud(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    LEFT = [True, False, False]
    RIGHT = [False, True, False]
    try:
        print('Note: positive error means enemy is to the RIGHT of screen center (x=80).')
        for episode in range(1, args.episodes + 1):
            game.new_episode()
            print(f'\nEpisode {episode}:')
            observations = [('START', enemy_info(game))]
            # Same episode: left 2, right 4, left 2 tics (approximately return).
            for action_name, action, duration in (
                ('MOVE_LEFT', LEFT, args.tics),
                ('MOVE_RIGHT', RIGHT, 2 * args.tics),
                ('MOVE_LEFT', LEFT, args.tics),
            ):
                if game.is_episode_finished():
                    break
                before = enemy_info(game)
                game.make_action(action, duration)
                after = enemy_info(game)
                observations.append((action_name, after))
                if before is not None and after is not None:
                    dx = after[0] - before[0]
                    print(f'  {action_name:10s} {fmt(before)} -> {fmt(after)} | delta_x={dx:+.1f}')
                else:
                    print(f'  {action_name:10s} {fmt(before)} -> {fmt(after)} | delta_x=unknown')
            print('  Initial:', fmt(observations[0][1]))
    finally:
        game.close()


if __name__ == '__main__':
    main()
