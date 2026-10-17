# Codex handoff: organize the repository without changing the science

## Purpose

This document describes a **proposed refactor**, not a refactor that has already been completed. The current repository stores an original, flat set of scripts under `CNN-build/`. The immediate objective for Codex is to improve structure, discoverability, CLI consistency, testing and packaging **while preserving the previously recorded experimental behavior**.

Read [METHODOLOGY.md](METHODOLOGY.md), [EXPERIMENTS.md](EXPERIMENTS.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md), and the repository root README before editing.

## Ground truth and guardrails

- **Do not assert that an SNN or FPGA deployment has been implemented.** The completed results are CNN-only.
- Do not silently modify action semantics (left/right/attack), the teacher alignment threshold, image preprocessing, reward configuration, duration classes `(2,4,8)`, model layers, training losses or experimental defaults.
- The **teacher uses object labels**, but demonstration images and CNN inference do **not** consume labels. Preserve that information boundary.
- Do not collapse `doom_dqn_baseline.py` and `train_imitation_cnn.py` into a single algorithm. DQN and supervised imitation are scientifically different.
- **Experiment 5A** = brightness evaluation; **5B** = feature maps/Grad-CAM; **5C** = brightness augmentation and verification. Maintain this naming consistently.
- Keep `CNN-build/runs/` baseline summaries and legacy research log intact. Avoid committing large checkpoints, `.npz` datasets, generated images or caches.
- Changes should occur on a dedicated branch or pull request. Require passing tests before switching entry points. Backward compatibility for existing commands is valuable.

## Current source inventory and suggested ownership

| Existing file | Role |
| --- | --- |
| `basic.py` | Initial ViZDoom exploration |
| `doom_cnn.py`, `labelled_buffer_cnn.py` | Earlier exploratory CNN/label experiments; inspect before classifying |
| `doom_dqn_baseline.py` | Reward-driven DQN baseline |
| `calibrate_aiming.py` | Motion/aiming calibration |
| `labelled_aiming_agent.py`, `labelled_aiming_agent_safe.py` | Teacher controller variants |
| `train_imitation_cnn.py` | Canonical imitation CNN, model class and RGB preprocessing |
| `train_imitation_cnn_brightness.py` | Canonical CNN v2 training |
| `evaluate_imitation_robustness.py` | CNN-only evaluation |
| `evaluate_brightness_robustness.py` | 5A and later brightness condition evaluation |
| `experiment_5b_feature_visualization.py` | Feature maps and Grad-CAM |
| `compare_brightness_models.py` | Analysis of brightness summaries |
| `verify_cnn_v2.py` | 3,000-episode CNN v2 verification |

## Suggested future package layout (not yet present)

```text
projectDOOM/
├── README.md
├── AGENTS.md
├── pyproject.toml or environment.yml
├── CNN-build/
│   ├── docs/
│   ├── scripts/          # Thin backward-compatible CLI wrappers
│   ├── doom_cnn/         # Shared preprocessing, game setup, policies and networks
│   ├── experiments/      # Explicit experiment configurations
│   ├── tests/            # Fast unit tests and optional ViZDoom smoke tests
│   └── runs/             # Historical small summaries; preserve paths/contents
└── SNN-build/            # Reserved for future SNN research
```

Choose directory names only after inspecting dependencies and import graphs. Avoid introducing a package name that conflicts with existing `doom_cnn.py`; resolve naming thoughtfully and incrementally.

## Refactor sequence

1. Inventory all scripts, imported modules, CLI arguments, output schemas and local dependencies. Identify exploratory vs canonical versions without deleting historical evidence.
2. Extract shared environment configuration, RGB preprocessing, action/duration mapping, network definitions and checkpoint loading into a small internal package, retaining wrappers for original CLI invocation.
3. Write tests for screen shape/normalization, action-index interpretation, duration mapping, CNN output shapes, brightness clipping, seed/split determinism, and CSV aggregation.
4. Add a reproducible dependency specification and `--help` smoke tests. Separate quick CPU unit tests from optional engine/GPU integration tests.
5. Preserve and cross-reference old run histories and checkpoint provenance. Validate that old checkpoint `state_dict` keys still load exactly into any relocated CNN.
6. Update README commands **after** the new CLIs are verified. Keep documentation accurate about incomplete SNN and FPGA work.

## Acceptance criteria

- All original experimental outcomes remain documented and unchanged.
- Import/test commands succeed from the chosen repository root (and existing legacy commands are either maintained or explicitly deprecated).
- Old CNN v1/v2 checkpoint format remains readable without architecture changes.
- No dataset files, checkpoints, raw frames or private machine settings accidentally enter Git history.
- New code does not enable the label buffer during CNN-only inference.
- The final PR explains moved files, new commands, tests run, and any deviations from the baseline.

## A ready-to-use Codex task

> Inspect this repository and reorganize its existing CNN research scripts into a maintainable, testable Python project. Use README.md and CNN-build/docs/{METHODOLOGY,EXPERIMENTS,REPRODUCIBILITY,CODEX_HANDOFF}.md as the scientific source of truth. Preserve experiment names (5A/5B/5C), all reported CSV/JSON summaries, CNN architecture and checkpoint compatibility, label-free CNN inference, exact preprocessing/action mapping, and legacy command usability. Do not train new models or change experimental results. First produce an inventory and a proposed migration plan, then implement incrementally with CPU smoke tests and a PR. SNN-build is future work and must not be presented as completed.
