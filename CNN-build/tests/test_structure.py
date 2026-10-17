"""Dependency-light checks of the structural move and original pure functions."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
CNN = ROOT / "CNN-build"
MANIFEST = json.loads((Path(__file__).with_name("structural_baseline.json")).read_text())
try:
    import numpy as np
except ImportError:
    np = None


def script_path(name):
    return ROOT / next(e["new"] for e in MANIFEST["scripts"] if Path(e["old"]).stem == name)


def original_function(script, name, **namespace):
    """Execute an existing function unchanged, without importing engine/ML packages."""
    tree = ast.parse(script_path(script).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script_path(script)), "exec"), namespace)
    return namespace[name]


def defaults(script):
    tree = ast.parse(script_path(script).read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for kw in node.keywords:
                if kw.arg == "default":
                    result[ast.literal_eval(node.args[0])] = eval(
                        compile(ast.Expression(kw.value), "<default>", "eval"), {"Path": Path})
    return result


class StructureTests(unittest.TestCase):
    def test_all_original_source_preserved_except_import_paths(self):
        for entry in MANIFEST["scripts"]:
            with self.subTest(script=entry["old"]):
                source = (ROOT / entry["new"]).read_text(encoding="utf-8")
                for old, new in MANIFEST["import_moves"].items():
                    source = source.replace(f"from {new} import", f"from {old} import")
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), entry["sha256"])

    def test_historical_documentation_and_results_preserved(self):
        for path, digest in MANIFEST["protected_files"].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode()).hexdigest(), digest)

    def test_every_old_entry_point_exists(self):
        for entry in MANIFEST["scripts"]:
            source = (ROOT / entry["old"]).read_text(encoding="utf-8")
            self.assertIn(entry["module"], source)
            ast.parse(source)

    def test_package_imports_resolve_to_existing_files(self):
        for path in (CNN / "cnn_research").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("cnn_research."):
                    self.assertTrue((CNN / Path(*node.module.split("."))).with_suffix(".py").is_file())

    def test_artifact_ignore_rules(self):
        paths = ["CNN-build/runs/new/best.pt", "CNN-build/runs/new/demo_00001.npz",
                 "CNN-build/runs/experiment_5b_features_new/frame_cam.png"]
        result = subprocess.run(["git", "check-ignore", *paths], cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(set(result.stdout.splitlines()), set(paths))

    def test_training_defaults(self):
        expected = {"--epochs": 50, "--batch-size": 64, "--lr": 1e-4, "--val-fraction": 0.2,
                    "--patience": 8, "--duration-weight": 0.4, "--seed": 42, "--threads": 2}
        for name in ("train_imitation_cnn", "train_imitation_cnn_brightness"):
            actual = defaults(name)
            for key, value in expected.items():
                self.assertEqual(actual[key], value)
        actual = defaults("train_imitation_cnn_brightness")
        self.assertEqual([actual[k] for k in ("--brightness-min", "--brightness-max", "--augment-prob")], [0.5, 1.5, 0.8])

    def test_teacher_and_evaluation_defaults(self):
        teacher = defaults("labelled_aiming_agent_safe")
        self.assertEqual([teacher[k] for k in ("--tolerance", "--far-threshold", "--near-tics", "--far-tics", "--attack-tics", "--search-tics")], [4.0, 12.0, 2, 8, 4, 4])
        self.assertEqual(defaults("evaluate_brightness_robustness")["--brightness"], [1.0, 0.5, 0.75, 1.25, 1.5])
        verify = defaults("verify_cnn_v2")
        self.assertEqual((verify["--normal-episodes"], verify["--dark-episodes"]), (1000, 1000))

    def test_action_duration_mapping(self):
        tree = ast.parse(script_path("train_imitation_cnn").read_text(encoding="utf-8"))
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(values["ACTIONS"], ((True, False, False), (False, True, False), (False, False, True)))
        self.assertEqual(values["ACTION_NAMES"], ("LEFT", "RIGHT", "ATTACK"))
        self.assertEqual(values["TIC_CLASSES"], (2, 4, 8))

    def test_teacher_thresholds_and_search(self):
        choose = original_function("labelled_aiming_agent_safe", "choose_action", SCREEN_W=160)
        args = SimpleNamespace(tolerance=4, far_threshold=12, near_tics=2, far_tics=8, attack_tics=4, search_tics=4)
        for error, action, tics in [(-13, 0, 8), (-12, 0, 2), (-4, 2, 4), (4, 2, 4), (12, 1, 2), (13, 1, 8)]:
            enemy = SimpleNamespace(x=80 + error - 5, width=10)
            self.assertEqual(choose(enemy, args, 1), (action, tics, error, 1))
        self.assertEqual(choose(None, args, 1), (1, 4, None, -1))
        self.assertEqual(choose(None, args, -1), (0, 4, None, 1))

    def test_summary_analysis_cli_preserves_committed_comparison(self):
        entry = next(e for e in MANIFEST["scripts"] if Path(e["old"]).stem == "compare_brightness_models")
        with tempfile.TemporaryDirectory() as directory:
            for command in ([entry["old"].split("/")[-1]], ["-m", entry["module"]]):
                out = Path(directory) / ("legacy.csv" if len(command) == 1 else "module.csv")
                subprocess.run([sys.executable, *command, "--out", str(out)], cwd=CNN, check=True, capture_output=True)
                with out.open(newline="") as actual, (CNN / "runs/brightness_comparison_v1_vs_v2.csv").open(newline="") as historical:
                    self.assertEqual(list(csv.DictReader(actual)), list(csv.DictReader(historical)))


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


if __name__ == "__main__":
    unittest.main()
