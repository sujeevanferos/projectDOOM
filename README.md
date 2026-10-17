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
| DQN baseline | Investigated; unsatisfactory control in reported tests | `CNN-build/doom_dqn_baseline.py` |
| Label-guided aiming teacher | Completed | `CNN-build/labelled_aiming_agent_safe.py` |
| RGB imitation CNN v1 | Trained and evaluated | `CNN-build/train_imitation_cnn.py` |
| Experiment 5A: brightness stress test | Completed | `CNN-build/evaluate_brightness_robustness.py` |
| Experiment 5B: feature maps and Grad-CAM | Exploratory analysis completed | `CNN-build/experiment_5b_feature_visualization.py` |
| Experiment 5C: augmented CNN v2 | Trained, evaluated, and independently verified | `CNN-build/train_imitation_cnn_brightness.py`, `CNN-build/verify_cnn_v2.py` |
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

## Repository layout (before code reorganization)

```text
projectDOOM/
├── README.md                     # Research overview
├── .gitignore
└── CNN-build/
    ├── *.py                      # Original training, evaluation and teacher scripts
    ├── DOOM_CNN_research_log.md  # Earlier chronological notes
    ├── docs/                     # Methodology, evidence and Codex handoff
    └── runs/                     # Selected CSV/JSON summaries; large outputs ignored
```

Some historical paths contain the prefix `CNN-build/runs/`; scripts currently assume **the working directory is `CNN-build/`**. This repository is intentionally not yet reorganized, to preserve the scripts' original behavior.

## Getting started

Use a Linux environment with Python and ViZDoom, NumPy, PyTorch, and (for visualization) Matplotlib. Exact historical package versions are not pinned in this repository. ViZDoom must be able to locate its installed `basic.wad` scenario. The scripts support CUDA when available, and several support `--cpu`.

```bash
cd CNN-build
python labelled_aiming_agent_safe.py --episodes 1000 --headless --save-demos
python train_imitation_cnn.py --epochs 50
python train_imitation_cnn.py --eval --episodes 100 --headless
python train_imitation_cnn_brightness.py --epochs 50
python evaluate_brightness_robustness.py \
  --checkpoint runs/imitation_cnn_brightness/best.pt \
  --out-dir runs/brightness_robustness_v2_new --episodes 100 --headless
python verify_cnn_v2.py --headless --out-dir runs/cnn_v2_verification_new
```

**Do not rerun collection/training into existing results directories.** Several scripts explicitly refuse to overwrite saved outputs. Demonstration `.npz` files and trained `.pt` checkpoints are excluded from the Git repository, so a fresh clone cannot run evaluation without separately obtaining or recreating those artifacts. See [Reproducibility](CNN-build/docs/REPRODUCIBILITY.md).

## Documentation

- [Methodology and model architecture](CNN-build/docs/METHODOLOGY.md)
- [Experiment chronology and measurements](CNN-build/docs/EXPERIMENTS.md)
- [Reproduction and artifact management](CNN-build/docs/REPRODUCIBILITY.md)
- [Codex organization handoff](CNN-build/docs/CODEX_HANDOFF.md)
- [Legacy research log](CNN-build/DOOM_CNN_research_log.md)

## Limitations and planned work

The task is restricted to a single ViZDoom scenario, action vocabulary, and enemy class. The teacher provides privileged information **only during data generation**, and teacher-labeled episodes are not a substitute for diverse unseen maps or settings. Brightness modifications affect the **CNN input tensor**, not actual scene lighting or game physics. Grad-CAM provides qualitative attributions, not proof of object detection. SNN architecture, spike coding, surrogate-gradient training, and FPGA resource measurements remain future work.

## Citation and attribution

This repository is an evolving research prototype and is **not a peer-reviewed publication**. If citing the implementation, cite the GitHub repository and the specific commit used, together with the underlying ViZDoom and PyTorch tools. A formal bibliography, software version lockfile, and external scenario evaluations remain to be prepared.
