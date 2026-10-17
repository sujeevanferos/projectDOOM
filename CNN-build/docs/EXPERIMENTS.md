# Experimental record and evidence ledger

**Source mapping:** Short filenames in the timeline identify experiments, not executable root-level scripts. All canonical implementations reside in `CNN-build/cnn_research/`; see [the module mapping](MIGRATION.md#commands-and-imports). This record distinguishes committed artifacts from claims that require unavailable checkpoints or datasets.

**Evidence policy:** The figures below are transcribed from committed CSV/JSON summaries where available, or identified as previously reported measurements. A closed-loop success rate is distinct from supervised action accuracy. We use `basic.wad` unless explicitly noted.

## Timeline

| Stage | Script(s) | Question and outcome |
| --- | --- | --- |
| DQN baseline | `doom_dqn_baseline.py` | Could reward-only CNN control solve the task? Reported policies were unstable/collapsed. |
| Teacher localization | `calibrate_aiming.py`, `labelled_aiming_agent_safe.py` | Can privileged monster labels generate a consistent aiming policy? Yes, for the constrained scenario. |
| RGB imitation v1 | `train_imitation_cnn.py` | Can the CNN imitate teacher action/duration from pixels? High held-out teacher-label accuracy and closed-loop success. |
| 5A | `evaluate_brightness_robustness.py` | Does changing input brightness reveal a failure mode? Yes, particularly at 0.5× and 0.75×. |
| 5B | `experiment_5b_feature_visualization.py` | What do intermediate activations and action attributions suggest? Spatial patterns and brightness-sensitive predictions; qualitative only. |
| 5C | `train_imitation_cnn_brightness.py`, `compare_brightness_models.py` | Can augmenting training intensity improve dark-condition control? Improved observed kill rates and rewards in the initial comparison. |
| Verification | `verify_cnn_v2.py` | Do gains persist over a larger sample? 3,000/3,000 recorded kills for CNN v2 at three brightness conditions. |

## Dataset and supervised learning

The label-guided teacher produced **1,000 successful episodes**, **5,781 samples**, and mean episode reward **77.08** in the reported collection. The committed `runs/labelled_teacher/metrics.csv` records episode outcomes; the larger `.npz` demonstrations are ignored by Git.

| Metric at epoch 50 | CNN v1 | CNN v2 (brightness augmented) |
| --- | ---: | ---: |
| Train action accuracy | 99.4836% | 99.5697% |
| Validation action accuracy | 99.5587% | 99.7352% |
| Train duration accuracy | 91.8244% | 94.1050% |
| Validation duration accuracy | 91.0856% | 93.0274% |
| Validation total loss | 0.137786 | 0.105718 |

Evidence: `runs/imitation_cnn/history.csv` and `runs/imitation_cnn_brightness/history.csv`. All figures are supervised measures on frames from held-out demonstration episodes. Accuracy differences here are small and should not be conflated with generalization to new levels.

A previously reported 1,000-episode fresh CNN-v1 evaluation recorded **1,000/1,000 kills**, mean reward approximately **74.842**, and mean **6.371** decisions per episode. The original robust evaluation script is `evaluate_imitation_robustness.py`; its committed `summary.json` is not a full per-condition results table, so the narrative result should be treated separately from CSV-backed tables below.

## 5A: v1 brightness robustness

Each condition used 100 fresh, independent episodes. The model checkpoint was frozen, and the ViZDoom scene itself was unchanged; only CNN input intensity was scaled.

| Input multiplier | Kills | Mean reward | Mean decisions | Left / Right / Attack |
| --- | ---: | ---: | ---: | --- |
| 0.50× | 98/100 | 55.78 | 7.38 | 72 / 92 / 574 |
| 0.75× | 97/100 | 60.88 | 7.26 | 145 / 159 / 422 |
| 1.00× | 100/100 | 78.74 | 5.64 | 152 / 188 / 224 |
| 1.25× | 100/100 | 75.96 | 6.64 | 207 / 242 / 215 |
| 1.50× | 100/100 | 78.30 | 6.27 | 166 / 247 / 214 |

The larger attack counts at low brightness motivate the hypothesis that dark input leads to premature or repeated firing. These data do not establish the network's semantic failure mechanism.

Evidence: `runs/brightness_robustness/summary.csv` and `metadata.json`.

## 5B: CNN feature visualization

The qualitative study captured **9 RGB gameplay frames** from **5 episodes** and evaluated each image at 1.0× and 0.5× input intensity, giving 18 model evaluations. The paired-frame comparison showed changes in 4/9 action predictions (each movement → attack) and changes in 6/9 predicted durations. Intermediate feature maps exhibited progressively coarser spatial patterns. Grad-CAM maps were not uniquely concentrated on the enemy; some emphasized background regions as well.

These are exploratory attributions from a *small, nonrandom collection of frames*, not evidence of object detection or a statistically representative failure frequency. Visualization PNGs are intentionally not versioned.

Evidence: `runs/experiment_5b_features/predictions.csv`, `metadata.json`; images remain local to the experiment runner.

## 5C: CNN v2 brightness augmentation, 100 episodes/condition

| Input multiplier | v1 kills | v2 kills | v1 mean reward | v2 mean reward |
| --- | ---: | ---: | ---: | ---: |
| 0.50× | 98/100 | 100/100 | 55.78 | 70.94 |
| 0.75× | 97/100 | 100/100 | 60.88 | 77.20 |
| 1.00× | 100/100 | 100/100 | 78.74 | 74.82 |
| 1.25× | 100/100 | 100/100 | 75.96 | 78.04 |
| 1.50× | 100/100 | 100/100 | 78.30 | 75.86 |

The darker conditions improved by **+2** and **+3 percentage points** in observed kill rate, with mean reward improvements of **+15.16** and **+16.32**, respectively. Normal-brightness reward was **3.92** lower in this 100-episode comparison, despite unchanged 100/100 observed success. As trials were **unpaired**, these are descriptive differences, not matched-trial estimates.

Evidence: `runs/brightness_robustness/summary.csv`, `runs/brightness_robustness_v2/summary.csv`, and `runs/brightness_comparison_v1_vs_v2.csv`.

## Larger-scale CNN v2 verification: 3,000 episodes

| Brightness | Kills | Mean reward | Mean decisions | Minimum reward |
| --- | ---: | ---: | ---: | ---: |
| 1.00× | 1,000/1,000 | 76.131 | 6.189 | −29 |
| 0.50× | 1,000/1,000 | 76.100 | 6.204 | −62 |
| 0.75× | 1,000/1,000 | 75.685 | 6.277 | −81 |

The per-condition **95% Wilson interval** for 1,000 successes out of 1,000 is approximately **[99.617%, 100%]**. These experiments do not guarantee error-free future episodes. Negative minimum rewards show that a kill can still require inefficient or costly gameplay.

Evidence: `runs/cnn_v2_verification/summary.csv` and `metadata.json`. The verification uses fresh condition-specific random episodes; it is not a paired trial against CNN v1.

## Interpretation and threats to validity

1. **Constrained distribution:** one simple environment, one visible monster category, three discrete actions, and a deterministic teacher heuristic.
2. **Teacher dependence:** high classification accuracy reflects imitation of a strong rule-based controller. It does not establish general visual intelligence.
3. **Validation distribution:** episode-wise holdout prevents same-episode leakage but not similarity of scene states across episodes.
4. **Lighting intervention:** multiplied input pixels are not equivalent to physically different maps, shadows, sensors, or rendering pipelines.
5. **Unpaired estimates:** observed model differences may include random initial-state variation.
6. **Explainability:** Grad-CAM suggests decision-correlated regions, not verified causal object localization.
7. **Future hardware claims:** no SNN accuracy, spike-rate, throughput, power, timing, or FPGA resource measurements have been produced.

## Next research work

Preserve current checkpoints and original summary tables, lock the software environment, investigate additional perturbations in separately versioned experiments, then define an SNN specification and measurable FPGA targets. Avoid renumbering 5B (feature visualization) and 5C (brightness augmentation).
