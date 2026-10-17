# Methodology: ViZDoom visual imitation learning

**Code organization:** Canonical implementations are Python modules in `CNN-build/cnn_research/`. Historical short script names in this document refer to those modules; see [the migration mapping](MIGRATION.md#commands-and-imports). This methodology records the reported research design, not an independent rerun.

## Experimental environment and observation pipeline

The current completed CNN studies use the ViZDoom `basic.wad` scenario and `map01`. Scripts configure an RGB24 screen at 160 × 120, an episode timeout of 200 tics, start time 10, and living reward −1. Actions are `MOVE_LEFT`, `MOVE_RIGHT`, and `ATTACK`. The RGB screen is subsampled by a factor of two on each spatial dimension, yielding `uint8` training arrays of shape `(3, 60, 80)`; the student normalizes these to [0, 1].

The model's decisions are *macro-actions*: a categorical action paired with a categorical duration in `{2,4,8}` tics. Thus, the model does not choose a new action on every ViZDoom tic.

## Initial reinforcement-learning baseline

`doom_dqn_baseline.py` implements a basic deep Q-network with an online Q-network, target network, experience replay, epsilon-greedy control, and Huber temporal-difference loss. It uses the same CNN feature backbone but a single three-action Q-value head and typically executes each action for four tics. This is a distinct training paradigm from imitation learning: rewards and bootstrapped Q-targets, not demonstration action labels, drive DQN optimization.

The recorded baseline evaluation exhibited collapsed or inefficient behaviors; this motivated, but does not prove the general superiority of, the later imitation pipeline.

## Teacher controller and demonstrations

`labelled_aiming_agent_safe.py` enables ViZDoom's labels buffer and selects the largest visible monster label. For a monster box with horizontal origin `x` and width `w`, the teacher computes:

```text
horizontal_error = (x + w/2) - 160/2
```

The teacher fires if the absolute error is at most the default 4 pixels. Otherwise it moves left if the enemy lies left of the center, or right if the enemy lies right. It uses two-tic near movements, eight-tic far movements (beyond a default 12-pixel threshold), and four-tic attack/search actions. If no monster label is visible, it uses short alternating search moves. This steering direction was established empirically by the movement-calibration script.

Demonstration files `demo_*.npz` contain:

| Array | Purpose |
| --- | --- |
| `frames` | RGB image, `N × 3 × 60 × 80`, uint8 |
| `actions` | Integer indices for left/right/attack |
| `action_tics` | Number of game tics for each decision |
| `rewards` | Recorded transition/game rewards; not used as supervised imitation targets |

**Teacher labels and bounding boxes are not saved as student inputs.** The teacher may use privileged labels while generating action targets; the CNN subsequently uses RGB images only.

The recorded corpus includes 1,000 successful teacher episodes and 5,781 frame-action examples, with action counts 1,843 left, 1,876 right, and 2,062 attack.

## Student CNN architecture

`train_imitation_cnn.py` defines `DoomImitationCNN`:

```text
Input RGB (3 × 60 × 80)
  Conv2d(3,16,5,stride=2,pad=2) → ReLU → MaxPool2d(2)
  Conv2d(16,32,3,pad=1)        → ReLU → MaxPool2d(2)
  Conv2d(32,64,3,pad=1)        → ReLU
  AdaptiveAvgPool2d(3 × 4) → Flatten (768)
    ├── Action head: Linear(768,128) → ReLU → Linear(128,3)
    └── Duration head: Linear(768,64) → ReLU → Linear(64,3)
```

Training uses separate categorical cross-entropy losses:

```text
L_total = CE(action_logits, teacher_action)
        + 0.4 × CE(duration_logits, teacher_duration_class)
```

The code uses Adam with default learning rate `1e-4`, batch size `64`, seed `42`, 50 maximum epochs, and a validation-loss early-stopping patience of eight epochs. The demonstration episodes are shuffled with the seeded NumPy generator and split **by entire episode** into 80% training and 20% validation, preventing frames from a single demonstration episode from appearing in both partitions. The 200 validation episodes are from the same teacher and scenario distribution; this is not a new-map or new-enemy test.

The best checkpoint is selected using **validation total loss**, not just action accuracy.

## CNN v2: controlled training augmentation

`train_imitation_cnn_brightness.py` reuses `DoomImitationCNN`, the same demonstration corpus, identical train/validation split procedure, and the same optimization defaults. During training only, each image independently receives multiplicative brightness adjustment with probability `0.8`. The multiplier is drawn uniformly from `[0.5,1.5]` and image intensities are clipped to `[0,1]`. Remaining training images and all validation images are unmodified. CNN v2 is trained from scratch rather than fine-tuned on CNN v1.

This design varies one training factor while retaining the baseline architecture. The original checkpoint is stored separately and not modified.

## Evaluation and operational definitions

Imitation **action accuracy** is the fraction of validation frames for which `argmax(action_logits)` equals the teacher's action. **Duration accuracy** is the analogous agreement with one of three tic classes. Neither metric directly measures successful gameplay.

Closed-loop **kill success** is based on an increase in ViZDoom's `KILLCOUNT` during an episode. **Mean reward**, **mean decisions per episode**, and action counts describe efficiency and possible policy collapse. No demonstration labels, exploration, or gradient updates are used during CNN evaluation.

Brightness evaluations multiply the *preprocessed RGB input to the network*, not the simulated lighting conditions. Brightness groups and historical v1/v2 tests use fresh **unpaired** start states.

## Explainability limits

`experiment_5b_feature_visualization.py` captures CNN-generated frames and visualizes convolutional activations and Grad-CAM for selected actions. Such maps show associations between regions of a representation and selected scores. They cannot establish that the model correctly localizes or semantically recognizes an enemy, and their spatial coarseness limits interpretation.

## Reproducibility cautions

- The repository excludes large checkpoints and demonstration datasets; they must be recreated or restored locally.
- Script imports and default paths currently assume execution from `CNN-build/`.
- Exact package versions and random engine/version details are not frozen in the repository.
- Unpaired episodes and finite trials prohibit claims of perfect true success probability or robustness to arbitrary map/enemy perturbations.
