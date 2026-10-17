"""Standard-library checks of canonical entry points and preserved source."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
CNN = ROOT / "CNN-build"
MANIFEST = json.loads((Path(__file__).with_name("structural_baseline.json")).read_text())


def script_path(name):
    return ROOT / next(e["new"] for e in MANIFEST["scripts"] if Path(e["new"]).stem == name)


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
    def test_all_original_source_preserved_except_entry_point_updates(self):
        for entry in MANIFEST["scripts"]:
            with self.subTest(script=entry["old"]):
                source = (ROOT / entry["new"]).read_text(encoding="utf-8")
                updates = MANIFEST["entry_point_updates"].get(entry["new"], {})
                for old, new in reversed(list(updates.items())):
                    source = source.replace(new, old)
                for old, new in MANIFEST["import_moves"].items():
                    source = source.replace(f"from {new} import", f"from {old} import")
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), entry["sha256"])

    def test_historical_documentation_and_results_preserved(self):
        for path, digest in MANIFEST["protected_files"].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode()).hexdigest(), digest)

    def test_only_canonical_entry_points_exist(self):
        self.assertEqual(list(CNN.glob("*.py")), [])
        for entry in MANIFEST["scripts"]:
            self.assertFalse((ROOT / entry["old"]).exists())
            target = (CNN / Path(*entry["module"].split("."))).with_suffix(".py")
            self.assertEqual(target, ROOT / entry["new"])
            self.assertTrue(target.is_file())
            ast.parse(target.read_text(encoding="utf-8"))

    def test_no_imports_or_commands_depend_on_removed_wrappers(self):
        names = {Path(e["old"]).stem for e in MANIFEST["scripts"]}
        for path in (CNN / "cnn_research").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for name in names:
                self.assertNotIn(f"python {name}.py", source, str(path))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn(node.module.split(".")[0], names, str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name.split(".")[0], names, str(path))

    def test_brightness_working_directory_check_uses_canonical_path(self):
        tree = ast.parse(script_path("evaluate_brightness_robustness").read_text(encoding="utf-8"))
        expected = "cnn_research/training/train_imitation_cnn.py"
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and any(isinstance(value, ast.Constant) and value.value == expected for value in ast.walk(n.test)))
        # Evaluate the actual existence check relative to CNN-build without importing the evaluator.
        self.assertFalse(eval(compile(ast.Expression(guard.test), "<working-directory guard>", "eval"),
                              {"Path": lambda value: CNN / value}))

    def test_python_syntax_without_imports(self):
        for path in CNN.rglob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_current_documentation_links_exist(self):
        for path in (ROOT / "README.md", CNN / "docs/MIGRATION.md"):
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if not link.startswith(("http:", "https:")):
                    self.assertTrue((path.parent / link.split("#")[0]).exists(), f"{path}: {link}")

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
        entry = next(e for e in MANIFEST["scripts"] if Path(e["new"]).stem == "compare_brightness_models")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "comparison.csv"
            subprocess.run([sys.executable, "-S", "-m", entry["module"], "--out", str(out)],
                           cwd=CNN, check=True, capture_output=True)
            with out.open(newline="") as actual, (CNN / "runs/brightness_comparison_v1_vs_v2.csv").open(newline="") as historical:
                self.assertEqual(list(csv.DictReader(actual)), list(csv.DictReader(historical)))


if __name__ == "__main__":
    unittest.main()
