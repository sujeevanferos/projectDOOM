"""CPU unit tests; engine, original artifacts and gameplay have separate opt-ins."""
import ast
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from test_structure import CNN, MANIFEST, ROOT, original_function, script_path

try:
    import numpy as np
except ImportError:
    np = None

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_VIZDOOM = importlib.util.find_spec("vizdoom") is not None
sys.path.insert(0, str(CNN))


@unittest.skipUnless(HAS_TORCH and np is not None, "Install PyTorch and NumPy for CPU model tests")
class CPUModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        # Only unit tests use this import placeholder; no engine behavior is simulated here.
        # Real gameplay tests below require the actual ViZDoom installation.
        with patch.dict(sys.modules, {} if HAS_VIZDOOM else {"vizdoom": ModuleType("vizdoom")}):
            cls.student = importlib.import_module("cnn_research.training.train_imitation_cnn")
            cls.brightness = importlib.import_module("cnn_research.training.train_imitation_cnn_brightness")
            cls.evaluator = importlib.import_module("cnn_research.evaluation.evaluate_brightness_robustness")
            cls.verify = importlib.import_module("cnn_research.evaluation.verify_cnn_v2")

    def test_model_shapes_and_parameter_names(self):
        model = self.student.DoomImitationCNN().cpu().eval()
        with self.torch.inference_mode():
            action, duration = model(self.torch.zeros(2, 3, 60, 80))
        self.assertEqual(tuple(action.shape), (2, 3))
        self.assertEqual(tuple(duration.shape), (2, 3))
        expected = {"features.0.weight": (16, 3, 5, 5), "features.0.bias": (16,),
                    "features.3.weight": (32, 16, 3, 3), "features.3.bias": (32,),
                    "features.6.weight": (64, 32, 3, 3), "features.6.bias": (64,),
                    "action_head.0.weight": (128, 768), "action_head.0.bias": (128,),
                    "action_head.2.weight": (3, 128), "action_head.2.bias": (3,),
                    "duration_head.0.weight": (64, 768), "duration_head.0.bias": (64,),
                    "duration_head.2.weight": (3, 64), "duration_head.2.bias": (3,)}
        self.assertEqual({key: tuple(value.shape) for key, value in model.state_dict().items()}, expected)

    def test_baseline_model_state_dict_loads_into_relocated_model(self):
        entry = next(e for e in MANIFEST["scripts"] if Path(e["new"]).stem == "train_imitation_cnn")
        source = subprocess.check_output(["git", "show", f'{MANIFEST["baseline_commit"]}:{entry["old"]}'], cwd=ROOT, text=True, encoding="utf-8")
        namespace = {"__name__": "original_student"}
        with patch.dict(sys.modules, {} if HAS_VIZDOOM else {"vizdoom": ModuleType("vizdoom")}):
            exec(compile(source, "<baseline student>", "exec"), namespace)
        original = namespace["DoomImitationCNN"]().cpu().eval()
        relocated = self.student.DoomImitationCNN().cpu().eval()
        # Synthetic weights from the original definition; never saved to research runs.
        buffer = io.BytesIO()
        self.torch.save({"model": original.state_dict(), "tic_classes": (2, 4, 8), "architecture": "DoomImitationCNN_v1"}, buffer)
        buffer.seek(0)
        checkpoint = self.torch.load(buffer, map_location="cpu", weights_only=False)
        relocated.load_state_dict(checkpoint["model"], strict=True)
        images = self.torch.linspace(0, 1, 3 * 60 * 80).reshape(1, 3, 60, 80)
        with self.torch.inference_mode():
            for before, after in zip(original(images), relocated(images)):
                self.torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_package_consumers_use_same_model(self):
        self.assertIs(self.brightness.DoomImitationCNN, self.student.DoomImitationCNN)
        self.assertIs(self.evaluator.DoomImitationCNN, self.student.DoomImitationCNN)
        self.assertIs(self.verify.DoomImitationCNN, self.student.DoomImitationCNN)

    def test_dataset_rgb_normalization_and_duration_classes(self):
        episode = {"frames": np.full((3, 3, 60, 80), 255, dtype=np.uint8),
                   "actions": np.array([0, 1, 2]), "action_tics": np.array([2, 4, 8])}
        dataset = self.student.FrameDataset([episode])
        for index in range(3):
            image, action, duration = dataset[index]
            self.assertEqual(tuple(image.shape), (3, 60, 80))
            self.assertEqual(image.dtype, self.torch.float32)
            self.assertEqual((image.min().item(), image.max().item()), (1, 1))
            self.assertEqual((action, duration), (index, index))

    def test_brightness_augmentation_clipping_and_unchanged_targets(self):
        image = self.torch.tensor([0.0, 0.5, 1.0])
        base = [(image, 1, 2)]
        data = self.brightness.BrightnessTrainingDataset(base)
        with patch.object(self.brightness.random, "random", return_value=0.2), patch.object(self.brightness.random, "uniform", return_value=1.5):
            changed, action, duration = data[0]
        self.torch.testing.assert_close(changed, self.torch.tensor([0, 0.75, 1]))
        self.assertEqual((action, duration), (1, 2))
        self.torch.testing.assert_close(image, self.torch.tensor([0.0, 0.5, 1.0]))
        with patch.object(self.brightness.random, "random", return_value=0.8), patch.object(self.brightness.random, "uniform") as draw:
            self.assertIs(data[0][0], image)
            draw.assert_not_called()
        with patch.object(self.brightness.random, "random", return_value=0), patch.object(self.brightness.random, "uniform", return_value=0.5):
            self.torch.testing.assert_close(data[0][0], self.torch.tensor([0, 0.25, 0.5]))

    def test_evaluation_brightness_and_csv_schema_with_game_double(self):
        # This tests the actual evaluation transform/control wiring, not ViZDoom gameplay.
        frame = np.full((3, 120, 160), 201, dtype=np.uint8)
        class GameDouble:
            finished = False
            def new_episode(self): self.finished = False
            def is_episode_finished(self): return self.finished
            def get_state(self): return SimpleNamespace(screen_buffer=frame)
            def get_game_variable(self, _variable): return int(self.finished)
            def make_action(self, action, tics):
                self.action, self.tics = action, tics
                self.finished = True
                return 7.0
            def close(self): self.closed = True
        class CaptureModel:
            def __call__(self, images):
                self.images = images.clone()
                return torch.tensor([[0, 0, 1]]), torch.tensor([[0, 1, 0]])
        torch = self.torch
        for multiplier in (0.5, 1.5):
            game, model = GameDouble(), CaptureModel()
            with patch.object(self.evaluator, "create_game", return_value=game), patch.object(self.evaluator, "vzd", SimpleNamespace(GameVariable=SimpleNamespace(KILLCOUNT="kill"))):
                rows = self.evaluator.evaluate_condition(model, torch.device("cpu"), multiplier, 1, True)
            expected = min(201 * multiplier, 255) / 255
            self.assertAlmostEqual(model.images[0, 0, 0, 0].item(), expected, places=6)
            self.assertEqual((game.action, game.tics), ([False, False, True], 4))
            self.assertTrue(game.closed)
            self.assertEqual(list(rows[0]), ["brightness", "episode", "reward", "kills", "success", "decisions", "move_left", "move_right", "attack", "tics_2", "tics_4", "tics_8"])
            self.assertEqual(rows[0]["success"], 1)

    def test_synthetic_demo_loading_requires_rgb_only(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 2):
                np.savez(Path(directory) / f"demo_{index:05d}.npz", frames=np.zeros((1, 3, 60, 80), dtype=np.uint8),
                         actions=[index - 1], action_tics=[2], rewards=[1], enemy_labels=["unused"])
            episodes = self.student.load_demos(Path(directory))
            self.assertEqual(len(episodes), 2)
            self.assertEqual(set(episodes[0]), {"frames", "actions", "action_tics"})


@unittest.skipUnless(HAS_TORCH and HAS_VIZDOOM and importlib.util.find_spec("matplotlib"), "Install full requirements for real import/CLI help smoke tests")
class CLITests(unittest.TestCase):
    def test_all_argument_driven_entry_points(self):
        for entry in MANIFEST["scripts"]:
            if Path(entry["new"]).stem == "basic":
                continue  # Original script has no argparse and immediately starts visible gameplay.
            with self.subTest(module=entry["module"]):
                result = subprocess.run([sys.executable, "-m", entry["module"], "--help"], cwd=CNN, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertNotIn("RuntimeWarning", result.stderr)


@unittest.skipIf(np is None, "NumPy unavailable")
class ArrayTests(unittest.TestCase):
    def test_preprocessing_layout_shape_values_and_contiguity(self):
        chw = np.arange(3 * 120 * 160, dtype=np.uint8).reshape(3, 120, 160)
        for script, name in [("train_imitation_cnn", "preprocess"), ("doom_dqn_baseline", "preprocess"),
                             ("labelled_aiming_agent_safe", "rgb_chw_uint8")]:
            preprocess = original_function(script, name, np=np, SCREEN_W=160, SCREEN_H=120)
            for frame in [chw, chw.transpose(1, 2, 0)]:
                result = preprocess(frame)
                self.assertEqual(result.shape, (3, 60, 80))
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(result.flags.c_contiguous)
                np.testing.assert_array_equal(result, chw[:, ::2, ::2])
            with self.assertRaises(ValueError):
                preprocess(np.zeros((60, 80, 3), dtype=np.uint8))

    def test_episode_split_executes_original_statements(self):
        splits = []
        for script in ("train_imitation_cnn", "train_imitation_cnn_brightness"):
            tree = ast.parse(script_path(script).read_text(encoding="utf-8"))
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            # Execute only the original split assignments; no training or data loading.
            names = {"indices", "n_val", "val_indices", "train_indices", "val_eps", "train_eps"}
            nodes = [n for n in main.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
            namespace = {"np": np, "args": SimpleNamespace(seed=42, val_fraction=0.2), "episodes": list(range(1000))}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(script_path(script)), "exec"), namespace)
            train, val = namespace["train_eps"], namespace["val_eps"]
            self.assertEqual((len(train), len(val)), (800, 200))
            self.assertFalse(set(train) & set(val))
            self.assertEqual(set(train) | set(val), set(range(1000)))
            splits.append((train, val))
        self.assertEqual(splits[0], splits[1])
        manifest = json.loads((CNN / "runs/imitation_cnn_brightness/split_manifest.json").read_text())
        self.assertEqual([f"demo_{i + 1:05d}.npz" for i in splits[1][1]], manifest["validation_files"])


@unittest.skipUnless(HAS_TORCH and HAS_VIZDOOM and os.environ.get("DOOM_RUN_GAMEPLAY") == "1", "Optional: set DOOM_RUN_GAMEPLAY=1 with ViZDoom installed")
class GameplayTests(unittest.TestCase):
    def test_real_headless_environment(self):
        student = importlib.import_module("cnn_research.training.train_imitation_cnn")
        game = student.create_game(False)
        try:
            game.new_episode()
            frame = student.preprocess(game.get_state().screen_buffer)
            self.assertEqual(frame.shape, (3, 60, 80))
            self.assertIsNone(game.get_state().labels_buffer)
            game.make_action(list(student.ACTIONS[0]), 2)
        finally:
            game.close()


@unittest.skipUnless(HAS_TORCH and HAS_VIZDOOM and os.environ.get("DOOM_CHECKPOINT_DIR"), "Optional: set DOOM_CHECKPOINT_DIR to restored original runs directory")
class OriginalCheckpointTests(unittest.TestCase):
    def test_original_v1_v2_checkpoints(self):
        import torch
        student = importlib.import_module("cnn_research.training.train_imitation_cnn")
        for run in ("imitation_cnn", "imitation_cnn_brightness"):
            with self.subTest(run=run):
                path = Path(os.environ["DOOM_CHECKPOINT_DIR"]) / run / "best.pt"
                self.assertTrue(path.is_file(), f"Missing requested original checkpoint: {path}")
                model = student.DoomImitationCNN().cpu()
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model"], strict=True)
                self.assertEqual(tuple(checkpoint["tic_classes"]), (2, 4, 8))


if __name__ == "__main__":
    unittest.main()
