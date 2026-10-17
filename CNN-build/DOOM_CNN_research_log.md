# ViZDoom CNN research log

## Research goal
Learn reliable gameplay control of `basic.wad` from rendered frames. Keep the CNN-only solution independent of privileged object labels at evaluation.

## Experiment history (through October 17, 2026)

1. **Reward-only DQN baseline.** RGB 160×120 rendered frames, preprocessed to 3×60×80; three actions (left, right, attack); CNN with three convolutional blocks; DQN with replay buffer and target network. Trained for 1,000 episodes. Evaluation exposed collapsed policies (mostly shooting or moving right). One 20-episode evaluation of the episode-1000 checkpoint returned a mean reward of −129.40 with 5 positive-reward episodes. This is not a verified kill rate.
2. **Visual aiming idea.** Proposed distinguishing the enemy from the wall/background, aligning the weapon, then firing. Investigation of official ViZDoom API identified `state.labels` / `state.labels_buffer`, which provide visible-object categories and bounding boxes.
3. **Enemy localization.** ViZDoom identified a `Cacodemon` (`Monster`) and player (`Self`) with bounding boxes at 160×120 resolution. Enemy horizontal target position defined as `label.x + label.width / 2`; aiming center at x=80.
4. **Movement calibration.** Short MOVE_LEFT movements shifted the enemy rightward in the image; MOVE_RIGHT shifted it leftward. This established the controller action mapping. Changes with 2-tic actions were often only 0–1 pixel.
5. **Label-guided teacher.** Rule-based control using monster box center, alignment thresholds and variable-duration actions; 100/100 reported successful episodes, then 1,000/1,000 successful episodes. The 1,000-episode run had mean reward +77.08 and 5,781 recorded demonstrations: LEFT 1,843; RIGHT 1,876; ATTACK 2,062.
6. **Imitation learning.** CNN trained from 1,000 successful episodes, split by episode; outputs separate action and action-duration predictions. At epoch 50: validation action accuracy 99.56%; duration accuracy 91.09%. Label data not fed to CNN.
7. **CNN-only gameplay evaluation.** On 100 new episodes, recorded 100/100 kills, mean reward +76.28 and 6.29 decisions per episode. Agent predicts from RGB only; no privileged object labels at evaluation. Sample size remains finite and does not establish generalization to other scenarios.

## Next experiment: held-out robustness evaluation
- Run the **unchanged** imitation checkpoint for **1,000 fresh episodes**, no training, with labels buffer disabled.
- Save episode CSV, aggregate JSON summary and checkpoint metadata in a new output directory; do not overwrite the initial 100-episode evaluation.
- Report kill rate with a 95% Wilson interval, mean/median/min/max reward, decisions and action distributions.
- If successful, probe robustness across multiple independent runs and changed scenario settings one factor at a time, carefully identifying whether those settings differ from training.

## Reproducibility checklist
Preserve `doom_dqn_baseline.py`, `labelled_aiming_agent_safe.py`, `train_imitation_cnn.py`, `best.pt`, teacher demonstrations, CSV results, software environment export and any chosen random seeds. Record hardware and ViZDoom version for the final report.
