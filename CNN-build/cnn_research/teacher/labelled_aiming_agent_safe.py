#!/usr/bin/env python3
"""Label-guided ViZDoom basic.wad aiming teacher (NO neural-network training).

Examples:
    python labelled_aiming_agent.py --episodes 100 --headless
    python labelled_aiming_agent.py --episodes 1000 --headless --save-demos
    python labelled_aiming_agent.py --episodes 5 --verbose

Requires: pip install vizdoom numpy

The teacher uses privileged ViZDoom monster labels to pick actions. Saved
training inputs are RGB frames only, not labels or object coordinates.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import vizdoom as vzd

ACTIONS = (
    (True, False, False),   # MOVE_LEFT
    (False, True, False),   # MOVE_RIGHT
    (False, False, True),   # ATTACK
)
ACTION_NAMES = ("LEFT", "RIGHT", "ATTACK")
SCREEN_W, SCREEN_H = 160, 120


def create_game(visible=False):
    game = vzd.DoomGame()
    game.set_doom_scenario_path(str(Path(vzd.scenarios_path) / "basic.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_available_buttons([
        vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK,
    ])
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_living_reward(-1)
    game.set_render_hud(False)
    game.set_render_weapon(True)  # Match the original baseline visuals.
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game


def get_enemy(state):
    """Return largest *visible* monster label, or None."""
    if state is None or state.labels is None:
        return None
    monsters = [l for l in state.labels if l.object_category == "Monster"]
    if not monsters:
        return None
    return max(monsters, key=lambda l: l.width * l.height)


def rgb_chw_uint8(screen):
    """Match existing CNN preprocessing: RGB -> CHW uint8 at 60x80."""
    array = np.asarray(screen)
    if array.shape == (3, SCREEN_H, SCREEN_W):
        return np.ascontiguousarray(array[:, ::2, ::2])
    if array.shape == (SCREEN_H, SCREEN_W, 3):
        return np.ascontiguousarray(array[::2, ::2, :].transpose(2, 0, 1))
    raise ValueError(f"Unexpected RGB frame shape: {array.shape}")


def choose_action(enemy, args, search_sign):
    """Return action index, tics, image x-error (or None), search direction."""
    if enemy is None:
        # No label: small search motion, not an invented target coordinate.
        return (1 if search_sign > 0 else 0), args.search_tics, None, -search_sign

    error = (enemy.x + enemy.width / 2.0) - SCREEN_W / 2.0
    # Conservative threshold: don't oscillate around the crosshair.
    if abs(error) <= args.tolerance:
        return 2, args.attack_tics, error, search_sign
    action = 1 if error > 0 else 0  # empirically calibrated strafe direction
    tics = args.far_tics if abs(error) > args.far_threshold else args.near_tics
    return action, tics, error, search_sign


def episode(game, args, ep_num):
    game.new_episode()
    start_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    count = [0, 0, 0]
    reward_total = 0.0
    error_before = []
    error_improvement = []
    samples = []
    search_sign = 1
    target_visible_steps = 0
    decisions = 0
    while not game.is_episode_finished():
        gs = game.get_state()
        if gs is None:
            break
        monster = get_enemy(gs)
        action, tics, error, search_sign = choose_action(monster, args, search_sign)
        if error is not None:
            target_visible_steps += 1
            error_before.append(abs(error))
        image = rgb_chw_uint8(gs.screen_buffer) if args.save_demos else None
        reward = float(game.make_action(list(ACTIONS[action]), int(tics)))
        reward_total += reward
        count[action] += 1
        decisions += 1
        finished = game.is_episode_finished()
        if not finished and error is not None and action != 2:
            next_monster = get_enemy(game.get_state())
            if next_monster is not None:
                new_error = (next_monster.x + next_monster.width / 2.0) - SCREEN_W / 2.0
                error_improvement.append(abs(error) - abs(new_error))
        if args.save_demos:
            samples.append((image, action, reward, int(tics)))
        if args.verbose:
            error_str = "missing" if error is None else f"{error:+.1f}"
            print(f" ep={ep_num:03d} decision={decisions:03d} enemy_error={error_str:>7} "
                  f"action={ACTION_NAMES[action]:>6} tics={tics} reward={reward:+.1f}")

    end_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    kills = max(0, end_kills - start_kills)
    success = kills > 0
    result = dict(
        episode=ep_num, reward=round(reward_total, 3), kills=kills,
        success=int(success), decisions=decisions,
        move_left=count[0], move_right=count[1], attack=count[2],
        target_visible_steps=target_visible_steps,
        avg_abs_error=round(float(np.mean(error_before)), 3) if error_before else "",
        avg_error_reduction=round(float(np.mean(error_improvement)), 3) if error_improvement else "",
    )
    if args.save_demos and success and samples:
        args.demo_dir.mkdir(parents=True, exist_ok=True)
        dest = args.demo_dir / f"demo_{ep_num:05d}.npz"
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite existing demonstration: {dest}")
        np.savez_compressed(
            dest,
            frames=np.stack([x[0] for x in samples]),
            actions=np.asarray([x[1] for x in samples], dtype=np.int64),
            rewards=np.asarray([x[2] for x in samples], dtype=np.float32),
            action_tics=np.asarray([x[3] for x in samples], dtype=np.int16),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--headless", action="store_true", help="Hide ViZDoom window")
    p.add_argument("--tolerance", type=float, default=4.0, help="Shoot when abs horizontal error <= value")
    p.add_argument("--far-threshold", type=float, default=12.0)
    p.add_argument("--far-tics", type=int, default=8)
    p.add_argument("--near-tics", type=int, default=2)
    p.add_argument("--attack-tics", type=int, default=4)
    p.add_argument("--search-tics", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/labelled_teacher/metrics.csv"))
    p.add_argument("--save-demos", action="store_true", help="Save frame/action sequences for successful episodes")
    p.add_argument("--demo-dir", type=Path, default=Path("runs/labelled_teacher/demos"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.episodes < 1 or args.tolerance < 0 or args.far_threshold <= args.tolerance:
        p.error("Require episodes >= 1 and 0 <= tolerance < far-threshold")
    if min(args.far_tics, args.near_tics, args.attack_tics, args.search_tics) < 1:
        p.error("All tic durations must be at least one")
    # Fail before starting gameplay if this run would replace prior results.
    # Start each collection in a new output directory or explicitly clear it first.
    if args.out.exists():
        p.error(f"Metrics already exist: {args.out}. Use a new --out / --demo-dir or clear the old run explicitly.")
    if args.save_demos and args.demo_dir.exists() and any(args.demo_dir.glob("demo_*.npz")):
        p.error(f"Demonstrations already exist in {args.demo_dir}. Use an empty directory to avoid overwrite.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    game = create_game(visible=not args.headless)
    results = []
    try:
        with args.out.open("w", newline="") as f:
            writer = None
            for n in range(1, args.episodes + 1):
                row = episode(game, args, n)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                results.append(row)
                print(f"Ep {n:3d} | reward {row['reward']:7.1f} | kills {row['kills']} "
                      f"| success {row['success']} | decisions {row['decisions']:2d} "
                      f"| actions [{row['move_left']}, {row['move_right']}, {row['attack']}]")
    finally:
        game.close()
    print(f"\nEpisodes: {len(results)}")
    print(f"Success rate (KILLCOUNT): {100 * np.mean([r['success'] for r in results]):.1f}%")
    print(f"Mean episode reward: {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean decisions: {np.mean([r['decisions'] for r in results]):.1f}")
    print(f"Metrics saved to: {args.out}")
    if args.save_demos:
        saved = sorted(args.demo_dir.glob("demo_*.npz")) if args.demo_dir.exists() else []
        expected = sum(r["success"] for r in results)
        total_frames = 0
        actions = np.zeros(3, dtype=np.int64)
        bad_files = []
        for path in saved:
            try:
                with np.load(path) as data:
                    frames = data["frames"]
                    labels = data["actions"]
                    rewards = data["rewards"]
                    tics = data["action_tics"]
                    if frames.ndim != 4 or frames.shape[1:] != (3, 60, 80) or not (len(frames) == len(labels) == len(rewards) == len(tics)) or np.any((labels < 0) | (labels > 2)):
                        bad_files.append(path.name)
                    else:
                        total_frames += len(labels)
                        actions += np.bincount(labels.astype(int), minlength=3)
            except Exception as exc:
                bad_files.append(f"{path.name}: {exc}")
        print(f"Successful demonstrations directory: {args.demo_dir}")
        print(f"Saved demonstration files: {len(saved)} / {expected} successful episodes")
        print(f"Frame-action samples: {total_frames}; class counts LEFT/RIGHT/ATTACK: {actions.tolist()}")
        if len(saved) != expected or bad_files:
            raise RuntimeError(f"Dataset verification failed: count mismatch or invalid files: {bad_files[:5]}")
        print("Dataset verification: PASS")


if __name__ == "__main__":
    main()
