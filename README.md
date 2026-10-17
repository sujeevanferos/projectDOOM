# projectDOOM: Vision-Based Control in ViZDoom

**Research-stage repository | CNN baseline for a future spiking neural network (SNN) and FPGA investigation**

## Abstract

This project investigates visual decision-making in the ViZDoom `basic.wad` environment as a preliminary stage toward implementing spiking neural networks (SNNs) on FPGA hardware. The current completed work concerns conventional convolutional neural networks (CNNs). An initial reward-driven deep Q-network (DQN) baseline exhibited unstable, sometimes degenerate policies. The subsequent pipeline used ViZDoom's object-label interface to construct a rule-based aiming teacher, collected successful gameplay demonstrations, and trained a CNN by supervised imitation using RGB frames alone. A second CNN, trained with stochastic brightness augmentation, was evaluated for robustness to changes in input intensity. In the available evaluations, this second model achieved 3,000 enemy eliminations in 3,000 trials across three brightness settings. These findings concern a constrained single-scenario benchmark, **not** general DOOM competence, calibrated recognition of the enemy, or an implemented SNN/FPGA system.

## Research questions

1. Can a CNN learn an effective left/right/fire controller from rendered RGB observations?
2. How does supervised imitation of a reliable, privileged-information teacher compare with the initial reward-only DQN baseline?
3. What spatial features and action-attribution patterns are observed inside the imitation CNN?
4. Does brightness augmentation improve closed-loop control under input-intensity perturbations?
5. Which aspects of the CNN baseline can guide subsequent SNN and hardware design?

## Scope and status

| Component | Status | Evidence |
| --- | --- | --- |
| DQN baseline | Investigated; unsatisfactory control in reported tests | `CNN-build/cnn_research/training/doom_dqn_baseline.py` |
| Label-guided aiming teacher | Completed | `CNN-build/cnn_research/teacher/labelled_aiming_agent_safe.py` |
| RGB imitation CNN v1 | Trained and evaluated | `CNN-build/cnn_research/training/train_imitation_cnn.py` |
| Experiment 5A: brightness stress test | Completed | `CNN-build/cnn_research/evaluation/evaluate_brightness_robustness.py` |
| Experiment 5B: feature maps and Grad-CAM | Exploratory analysis completed | `CNN-build/cnn_research/analysis/experiment_5b_feature_visualization.py` |
| Experiment 5C: augmented CNN v2 | Trained, evaluated, and independently verified | `CNN-build/cnn_research/training/train_imitation_cnn_brightness.py`, `CNN-build/cnn_research/evaluation/verify_cnn_v2.py` |
| SNN implementation and FPGA deployment | Future research | Not represented as a completed result |

## Method overview

The ViZDoom environment renders RGB24 frames at **160 × 120**, then the student network subsamples them to **3 × 60 × 80** (channel-first). The action vocabulary is `MOVE_LEFT`, `MOVE_RIGHT`, and `ATTACK`; a second prediction head chooses an execution duration from **2, 4, or 8 game tics**.

The teacher controller uses privileged bounding boxes to determine the enemy's horizontal offset from the image center, but its saved demonstration inputs contain RGB arrays, actions, rewards, and action durations—not bounding boxes. At CNN training and inference, enemy labels are **not** provided. The imitation CNN has three convolutional layers and two classification heads, optimized with cross-entropy. CNN v2 reuses the same model and episode split, applying multiplicative brightness augmentation only to the training images.

**Important distinction:** the label-guided controller is used for dataset generation; the trained CNN is evaluated from RGB alone. The reported imitation accuracy measures agreement with teacher actions, whereas gameplay success measures an enemy kill reported by ViZDoom's `KILLCOUNT`.

## Selected reported findings

| Evaluation | Recorded result | Interpretation |
| --- | --- | --- |
| Teacher demonstration collection | 1,000/1,000 kills; 5,781 frame–action samples | Successful teacher examples, not a CNN result |
| CNN v1, held-out demonstration frames | 99.56% action accuracy at epoch 50 | Teacher-action classification, not gameplay success |
| CNN v2, held-out demonstration frames | 99.74% action accuracy at epoch 50 | Same validation split procedure; augmentation on training only |
| CNN v1, 5A at 0.50× / 0.75× | 98/100 and 97/100 kills | Sensitivity to darker inputs |
| CNN v2, initial 5C at 0.50× / 0.75× | 100/100 and 100/100 kills | Improved observed dark-condition reliability |
| CNN v2, later verification at 1.0× / 0.50× / 0.75× | 1,000/1,000 kills **at each** intensity | 3,000/3,000 observed kills; finite-sample estimate |

For the 1,000-trial verification conditions, the 95% Wilson interval for each observed 100% kill rate has a lower bound of approximately **99.62%**. Mean rewards were **76.131** (1.0×), **76.100** (0.50×), and **75.685** (0.75×). Comparisons between models were made on **unpaired random episodes**; improvements should not be described as matched-seed causal estimates or universal guarantees.

Source data: `CNN-build/runs/labelled_teacher/metrics.csv`, `CNN-build/runs/imitation_cnn/history.csv`, `CNN-build/runs/imitation_cnn_brightness/history.csv`, and the committed brightness and verification `summary.csv` files.

## Repository layout

```text
projectDOOM/
├── README.md                     # Research overview
├── .gitignore
├── requirements.txt              # Imported dependencies; historical versions unknown
├── CNN-build/
│   ├── cnn_research/             # Original implementations, grouped by purpose
│   │   ├── exploration/          # basic.py, doom_cnn.py, labelled_buffer_cnn.py
│   │   ├── teacher/              # Calibration and both label-guided teachers
│   │   ├── training/             # DQN baseline, imitation CNN v1 and v2
│   │   ├── evaluation/           # Gameplay robustness, brightness, v2 verification
│   │   └── analysis/             # Feature maps / Grad-CAM, summary comparison
│   ├── tests/                    # Structural checks and optional runtime tests
│   ├── docs/                     # Research documentation; archive/ holds original log
│   └── runs/                     # Historical CSV/JSON evidence at original paths
└── SNN-build/
    └── README.md                 # Future research only
```

This is a **structural organization only**: existing functions, classes, model definitions, defaults and scientific behavior are preserved. Shared components and configurations have not been extracted. The canonical model, student preprocessing, environment setup and episode dataset remain together in `cnn_research.training.train_imitation_cnn`; the five downstream scripts import that module explicitly. SNN/FPGA work remains future research.

Experiment commands still require **the working directory to be `CNN-build/`**. All relative inputs and outputs keep their original meaning. Legacy `python script.py ...` invocations are no longer supported; use `python -m cnn_research.<area>.<script> ...`. Use the module interface rather than directly executing files inside the package. See the [complete old-to-new mapping and inspection findings](CNN-build/docs/MIGRATION.md).

## Getting started

Use an existing Linux environment with Python and the dependencies inventoried in [requirements.txt](requirements.txt): ViZDoom, NumPy, PyTorch, and Matplotlib. Exact historical package versions are unknown; this file is not an environment lock. Preserve your server's CUDA-compatible PyTorch installation. ViZDoom must locate its installed `basic.wad` scenario. CUDA selection and existing `--cpu` flags are unchanged. No libraries need to be installed for the repository moves themselves.

The following commands use fresh output directories. Their arguments were checked against the preserved scripts; **training, gameplay, visualization and original-checkpoint loading were not executed during this organization** because their dependencies/artifacts are unavailable here. CLI help checks are included in the test suite for an environment with the full dependencies.

```bash
cd CNN-build
python -m cnn_research.teacher.labelled_aiming_agent_safe \
  --episodes 1000 --headless --save-demos \
  --out runs/labelled_teacher_new/metrics.csv --demo-dir runs/labelled_teacher_new/demos
python -m cnn_research.training.train_imitation_cnn --epochs 50 \
  --demo-dir runs/labelled_teacher_new/demos --out-dir runs/imitation_cnn_new
python -m cnn_research.training.train_imitation_cnn --eval --episodes 100 --headless \
  --checkpoint runs/imitation_cnn_new/best.pt --out-dir runs/imitation_eval_new
python -m cnn_research.training.train_imitation_cnn_brightness --epochs 50 \
  --demo-dir runs/labelled_teacher_new/demos --out-dir runs/imitation_cnn_brightness_new
python -m cnn_research.evaluation.evaluate_brightness_robustness \
  --checkpoint runs/imitation_cnn_brightness_new/best.pt \
  --out-dir runs/brightness_robustness_v2_new --episodes 100 --headless
python -m cnn_research.analysis.experiment_5b_feature_visualization \
  --checkpoint runs/imitation_cnn_new/best.pt --out-dir runs/experiment_5b_features_new \
  --episodes 12 --samples 9 --brightness 1.0 0.5 --headless
python -m cnn_research.evaluation.verify_cnn_v2 --headless \
  --checkpoint runs/imitation_cnn_brightness_new/best.pt --out-dir runs/cnn_v2_verification_new
```

**Do not rerun collection/training into existing results directories.** Several scripts explicitly refuse to overwrite saved outputs. Demonstration `.npz` files and trained `.pt` checkpoints are excluded from the Git repository, so a fresh clone cannot run evaluation without separately obtaining or recreating those artifacts. See [Reproducibility](CNN-build/docs/REPRODUCIBILITY.md).

## Verification

From the repository root, using the existing Python environment:

```bash
python -m unittest discover -s CNN-build/tests -v
python -m compileall -q CNN-build
git diff --check
```

The organization was checked on Python 3.14.2: **12 tests passed, 10 skipped** (earlier structural migration; not a fresh test run for the current branch). Source-preservation hashes cover all 14 implementations (allowing only six import statements to change), and every existing research document and CSV/JSON evidence file under `CNN-build/`. Available tests also cover preprocessing, action/duration mappings, teacher thresholds, defaults, the original episode split against the committed manifest, and both summary-comparison CLI interfaces. The comparison reproduced the committed table into a temporary directory.

PyTorch-dependent CPU tests and full import/CLI help checks explicitly skip when their dependencies are missing. CPU tests include output shapes, RGB normalization, brightness transforms, and a strict load from synthetic weights created using the original model definition. These do not substitute for restoring original trained checkpoints. Real gameplay and original checkpoint tests require explicit opt-ins; see [verification details](CNN-build/docs/MIGRATION.md#verification-and-remaining-checks). No experiments were rerun and no trained artifacts were generated.

## Documentation

- [Methodology and model architecture](CNN-build/docs/METHODOLOGY.md)
- [Experiment chronology and measurements](CNN-build/docs/EXPERIMENTS.md)
- [Reproduction and artifact management](CNN-build/docs/REPRODUCIBILITY.md)
- [Legacy research log](CNN-build/docs/archive/DOOM_CNN_research_log.md)
- [Structural migration guide and verification](CNN-build/docs/MIGRATION.md)

The chronological research log is preserved as a historical record; its script names may refer to pre-migration paths. The migration guide records old-to-new mappings and structural verification. Methodology, experiments and reproducibility describe the current canonical package.

## Limitations and planned work

The task is restricted to a single ViZDoom scenario, action vocabulary, and enemy class. The teacher provides privileged information **only during data generation**, and teacher-labeled episodes are not a substitute for diverse unseen maps or settings. Brightness modifications affect the **CNN input tensor**, not actual scene lighting or game physics. Grad-CAM provides qualitative attributions, not proof of object detection. SNN architecture, spike coding, surrogate-gradient training, and FPGA resource measurements remain future work.

## Citation and attribution

This repository is an evolving research prototype and is **not a peer-reviewed publication**. If citing the implementation, cite the GitHub repository and the specific commit used, together with the underlying ViZDoom and PyTorch tools. A formal bibliography, software version lockfile, and external scenario evaluations remain to be prepared.
