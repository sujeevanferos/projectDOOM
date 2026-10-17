# Structural migration guide

This phase moves existing scripts into a small Python package and retains their
original entry points. It does not extract or rewrite functions, consolidate
algorithms, introduce new configurations, or rerun experiments. Stage 2 code
refactoring requires explicit authorization.

Baseline: commit `645c0ce4a023240b3776b0e78e7b0a41c0e011a5`. All research
documents, historical results and artifact paths under `CNN-build/` are preserved.
The new `SNN-build/README.md` reserves future work; it contains no implementation.

## Commands and imports

Continue to run experiments after `cd CNN-build`. Each old invocation remains
valid with the same arguments. In the table, replace `python OLD.py` with
`python -m MODULE`; append the exact same CLI arguments.

| Old script | New module / source location |
| --- | --- |
| `basic.py` | `cnn_research.exploration.basic` |
| `doom_cnn.py` | `cnn_research.exploration.doom_cnn` |
| `labelled_buffer_cnn.py` | `cnn_research.exploration.labelled_buffer_cnn` |
| `calibrate_aiming.py` | `cnn_research.teacher.calibrate_aiming` |
| `labelled_aiming_agent.py` | `cnn_research.teacher.labelled_aiming_agent` |
| `labelled_aiming_agent_safe.py` | `cnn_research.teacher.labelled_aiming_agent_safe` |
| `doom_dqn_baseline.py` | `cnn_research.training.doom_dqn_baseline` |
| `train_imitation_cnn.py` | `cnn_research.training.train_imitation_cnn` |
| `train_imitation_cnn_brightness.py` | `cnn_research.training.train_imitation_cnn_brightness` |
| `evaluate_imitation_robustness.py` | `cnn_research.evaluation.evaluate_imitation_robustness` |
| `evaluate_brightness_robustness.py` | `cnn_research.evaluation.evaluate_brightness_robustness` |
| `verify_cnn_v2.py` | `cnn_research.evaluation.verify_cnn_v2` |
| `experiment_5b_feature_visualization.py` | `cnn_research.analysis.experiment_5b_feature_visualization` |
| `compare_brightness_models.py` | `cnn_research.analysis.compare_brightness_models` |

For example, `python train_imitation_cnn.py --epochs 50` maps to
`python -m cnn_research.training.train_imitation_cnn --epochs 50`.
The [root README](../../README.md#getting-started) provides the complete teacher,
v1, v2, brightness, visualization and verification workflow with fresh paths.

Module names correspond to files below `CNN-build/` by replacing dots with `/`
and appending `.py`. Do not invoke those nested files directly: `python -m`
provides the package import context. The package is intentionally local to
`CNN-build/`; it is not installed globally and does not change the working
directory. Running from the repository root requires first entering `CNN-build/`.

Original function/class names and constants remain available through the legacy
wrappers, for example `from train_imitation_cnn import DoomImitationCNN`.
New callers can use the explicit package import
`from cnn_research.training.train_imitation_cnn import DoomImitationCNN`.
Definitions live in the package now; internal imports use that location directly.
Code relying on a definition's `__module__`, source inspection through wrappers,
or monkeypatching a wrapper's globals should target the implementation module.
Source filenames embedded in original docstrings, CLI messages and metadata are
retained, and refer to working legacy entry points.

## Responsibilities and dependencies

- `exploration/`: preserves earlier engine and CNN investigations. These have
  distinct configurations and are not substitutes for the canonical experiments.
- `teacher/`: calibration plus original and safe label-guided controller variants.
  The safe collector is the documented demonstration entry point.
- `training/`: preserves the reward-driven baseline and the two imitation
  trainers. Model, student environment setup, preprocessing, demonstration loading,
  losses and `FrameDataset` remain inside `train_imitation_cnn.py`.
- `evaluation/`: closed-loop robustness, 5A brightness evaluation and the later
  frozen v2 verification. Their metrics and output schemas are unchanged.
- `analysis/`: 5B features/Grad-CAM and comparison of existing summary tables.
- `tests/`: structural preservation manifest and dependency-aware verification.
- `docs/` and `runs/`: original research records at their original locations.

The v2 trainer, two robustness evaluators, verification and feature visualization
import the canonical imitation module. Verification additionally imports the
brightness evaluator. Only these six `from ... import ...` statements changed in
the moved source. No shared environment/model/configuration modules were created.
Duplicate teacher, preprocessing and DQN code intentionally remains until Stage 2.
The package name avoids collision with the exploratory `doom_cnn.py` entry point.

## Inspection findings retained without scientific changes

1. `exploration/doom_cnn.py` preprocesses full-resolution `(3,120,160)` images
   and uses Double DQN. The canonical `training/doom_dqn_baseline.py` subsamples
   to `(3,60,80)` and uses vanilla DQN. These historical variants must not be
   merged by treating them as identical.
2. `exploration/basic.py` renders at 640x480 and enables diagnostic buffers;
   general methodology descriptions of the 160x120 CNN pipeline do not describe
   this engine example. It starts visible gameplay immediately and has no
   argparse CLI, so `--help` is not a safe smoke test for that file.
3. `exploration/labelled_buffer_cnn.py` enables the label buffer and prints object
   diagnostics, while its CNN consumes RGB. Canonical student evaluation does
   not enable labels. The exploratory diagnostic script is not the teacher.
4. `analysis/experiment_5b_feature_visualization.py` converts multiplied/clipped
   brightness pixels to uint8 before normalization. The closed-loop brightness
   evaluator retains float32 values. Their transformations have deliberately
   not been unified because that would alter the experiments.
5. `docs/EXPERIMENTS.md` understates the available v1 1,000-episode JSON evidence:
   `runs/imitation_robustness_1000/summary.json` records 1,000 successes, mean
   reward 74.842 and mean decisions 6.371, among other aggregates. It has no
   per-episode rows or multi-brightness condition table. The original document
   is preserved; this finding records the discrepancy separately.
6. The original reproducibility and handoff documents describe the flat layout
   and propose extraction. Their legacy commands still work through wrappers;
   extraction is deferred. Existing script-level configuration defaults remain
   authoritative. No new experiment configuration files were introduced.

## Checkpoints, data and outputs

Both imitation trainers save dictionaries containing a `model` state_dict, with
unchanged parameter names, tensor shapes and architecture identifier
`DoomImitationCNN_v1`. v2 adds augmentation metadata. Moving the class does not
change these keys. Existing checkpoint loading code is unchanged and needs no
checkpoint conversion or retraining. Loading the actual trained v1/v2 files
remains an optional verification step because they are not in this checkout.

Keep restored `.pt` files and `.npz` demonstrations in their existing local
locations under `CNN-build/runs/`. All paths, split procedures, seeds, augmentation,
action indices/durations, CSV schemas and overwrite protections remain as before.
Never use original result directories for new runs. `requirements.txt` lists
actual imported third-party packages without inventing historical versions or
changing the CUDA installation. No dependencies were installed for this work.

`.gitignore` still excludes datasets and checkpoints and now excludes generated
PNG files throughout `CNN-build/runs/`, including newly named visualization runs.
Already tracked CSV/JSON evidence is preserved. No local research artifacts were
deleted or relocated.

## Verification and remaining checks

From the repository root:

```bash
python -m unittest discover -s CNN-build/tests -v
python -m compileall -q CNN-build
git diff --check
```

On the existing Python 3.14.2 environment: **22 tests discovered, 12 passed,
10 skipped**. NumPy and Matplotlib are available; PyTorch and ViZDoom are absent.
No library installation is required to organize the repository. No pre-existing
test suite, linter or build configuration was present.

The passing checks verify all moved source against normalized SHA-256 snapshots
from the baseline, reversing only the permitted import substitutions. They also
verify every original CNN research document and CSV/JSON file, legacy entry-point
presence, package import targets, ignored artifact patterns, actual preprocessing
function behavior, action/duration semantics, aiming thresholds, defaults and the
original split statements against the historical v2 manifest. Pure-function tests
compile selected original AST nodes unchanged to avoid importing absent engine/ML
packages. They do not claim full runtime import coverage.

Both old and new summary comparison CLIs were executed using the committed v1/v2
summary inputs; temporary outputs matched the committed comparison table exactly.
Model execution, augmentation, synthetic checkpoint compatibility and all 26
dependency-backed `--help` invocations are implemented as tests but skipped here.
The synthetic checkpoint test executes the baseline model definition from Git,
strict-loads its generated state_dict into the relocated model, and compares CPU
predictions. This requires retaining the baseline commit in local Git history.
One evaluation unit test uses an explicit game double to check the actual pixel
transformation and CSV schema; it is not evidence of ViZDoom gameplay.

On a configured Linux research server, run the same tests using the existing CUDA
environment; model unit tests explicitly use CPU. Then optionally run real engine
and restored-checkpoint checks:

```bash
DOOM_RUN_GAMEPLAY=1 python -m unittest discover -s CNN-build/tests -v
DOOM_CHECKPOINT_DIR="$PWD/CNN-build/runs" python -m unittest discover -s CNN-build/tests -v
```

The gameplay opt-in initializes a real headless student environment and takes one
action. It is not a robustness evaluation. The checkpoint opt-in requires **both**
original `imitation_cnn/best.pt` and `imitation_cnn_brightness/best.pt`; missing files
fail rather than silently pass. Neither optional test ran during organization.
The original demonstration corpus, real model evaluations, feature generation and
CUDA execution also require manual verification on the server. No gameplay,
training, new measurements, SNN research or FPGA work was performed here.
