# Reproducibility and artifact-management protocol

## Execution assumptions

Current implementations live under the `CNN-build/cnn_research/` package. The historical standalone entry-point files have been removed. Run commands after `cd CNN-build` using `python -m cnn_research.<area>.<module>` so package imports resolve correctly. A compatible environment requires Python, PyTorch, NumPy, ViZDoom (including the installed `basic.wad` asset), and Matplotlib for image/Grad-CAM plots. Exact library versions are **not locked** in the current repository; record them when rerunning.

Examples for a server environment:

```bash
python --version
python -c "import torch, numpy, vizdoom; print(torch.__version__, numpy.__version__, vizdoom.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Reproduction workflow (use unique output directories)

From `CNN-build/`:

```bash
# 1. Build successful teacher demonstrations
python -m cnn_research.teacher.labelled_aiming_agent_safe --episodes 1000 --headless --save-demos \
  --out runs/labelled_teacher_new/metrics.csv \
  --demo-dir runs/labelled_teacher_new/demos

# 2. Train original architecture with unaugmented examples
python -m cnn_research.training.train_imitation_cnn \
  --demo-dir runs/labelled_teacher_new/demos \
  --out-dir runs/imitation_cnn_new --epochs 50

# 3. Test CNN v1 with no label access
python -m cnn_research.training.train_imitation_cnn --eval --headless --episodes 100 \
  --checkpoint runs/imitation_cnn_new/best.pt \
  --out-dir runs/imitation_eval_new

# 4. Brightness stress test for CNN v1
python -m cnn_research.evaluation.evaluate_brightness_robustness --headless --episodes 100 \
  --checkpoint runs/imitation_cnn_new/best.pt \
  --out-dir runs/brightness_robustness_new

# 5. Train CNN v2 with the same demonstration source
python -m cnn_research.training.train_imitation_cnn_brightness --epochs 50 \
  --demo-dir runs/labelled_teacher_new/demos \
  --out-dir runs/imitation_cnn_brightness_new

# 6. Evaluate and verify CNN v2
python -m cnn_research.evaluation.evaluate_brightness_robustness --headless --episodes 100 \
  --checkpoint runs/imitation_cnn_brightness_new/best.pt \
  --out-dir runs/brightness_robustness_v2_new

python -m cnn_research.evaluation.verify_cnn_v2 --headless \
  --checkpoint runs/imitation_cnn_brightness_new/best.pt \
  --out-dir runs/cnn_v2_verification_new
```

**Caution:** `train_imitation_cnn.py --eval` writes its evaluation CSV under `--out-dir`, and it calls `mkdir` on that output path. Creating fresh output directories avoids collisions. Many scripts refuse to overwrite existing results; confirm each script's CLI with `--help` before automated batch execution.

## Feature visualization

```bash
python -m cnn_research.analysis.experiment_5b_feature_visualization \
  --checkpoint runs/imitation_cnn_new/best.pt \
  --out-dir runs/experiment_5b_features_new \
  --episodes 12 --samples 9 --brightness 1.0 0.5 --headless
```

Visualization is diagnostic. Saving only figures without frame IDs and brightness factors is insufficient for reproducing comparisons; preserve generated metadata and prediction tables.

## Inputs and outputs

| Artifact | Source | Version control |
| --- | --- | --- |
| `*.py`, `*.md` | Source, research notes | Commit |
| `runs/labelled_teacher/demos/demo_*.npz` | Teacher demonstrations | Retain locally/archival storage; excluded from Git |
| `runs/imitation_cnn/best.pt` | CNN v1 weights | Retain locally/archival storage; excluded from Git |
| `runs/imitation_cnn_brightness/best.pt` | CNN v2 weights | Retain locally/archival storage; excluded from Git |
| `runs/**/summary.csv`, `history.csv`, metadata | Aggregate evidence | Commit when small and free of sensitive data |
| `runs/**/episodes.csv`, plotted PNGs | Large per-episode/visual outputs | Usually excluded from Git |

The repository contains *selected* historical summary data, not a complete self-contained reproducibility package. Checkpoints and teacher demonstrations must be restored from trusted archival storage or regenerated with appropriate compute resources; the commands below are instructions, not evidence that a new run has been performed. `.gitignore` prevents newly added matching files from being staged but does not remove already tracked files.

## Minimum run manifest

For any new study, retain a machine-readable manifest with: script name and Git commit; experiment identifier; dataset provenance and split seed; software versions; random seeds; configuration/CLI arguments; machine/GPU; checkpoint SHA-256 and epoch; whether label access was enabled; whether gradient updates occurred; brightness multiplier and application point; episode counts; and whether conditions used paired start states.

Ensure `KILLCOUNT` is explicitly recorded separately from total reward. Do not describe a 100% observed kill rate as a proven 100% underlying probability.

## Research integrity safeguards

- Never overwrite original CNN v1/v2 checkpoints and original benchmark CSVs.
- Do not use label buffers as features for the trained student model; teacher-only access is intentional.
- Keep the episode-level split, and store the split manifest for CNN v2.
- Explain which data and model artifacts are missing from Git and how to restore them.
- For matched-condition comparisons, implement and verify reproducible start-state pairing rather than assuming it from a shared random seed.
- Before scientific reuse, validate imports, preprocessing shape, a synthetic model forward pass, teacher collection, evaluation and CSV schemas in a configured environment. A documentation change does not constitute an experimental replication.
