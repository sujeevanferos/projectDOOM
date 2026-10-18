# projectDOOM — Neuromorphic Control of DOOM

**Team research project | Visual game-playing agents → recurrent spiking neural networks → FPGA-based external control**

> **Research status (October 2026):** The CNN software baseline has been trained and extensively evaluated. The team has implemented RSNN prototypes and is currently training the newer RSNN v2. FPGA synthesis, hardware inference, and the proposed two-computer demonstration have **not yet been verified**.

## Vision

The goal of **projectDOOM** is to build an autonomous DOOM controller that runs as a **spiking neural network (SNN) on FPGA hardware**, receiving game observations from one computer and sending control actions back to that computer. The game runs on the host PC; the neural decision-making is intended to run on a separate physical FPGA platform. This will allow us to study learned control, spike-based computation, digital hardware constraints, and end-to-end communication latency.

The convolutional neural network (CNN) was deliberately developed as a **scientific and engineering baseline**, not as the final deliverable. It helped us understand how a learning-based controller interprets rendered frames, chooses gameplay actions without direct human input, and behaves under controlled visual perturbations. In parallel, the team is developing **recurrent spiking neural networks (RSNNs)** for eventual hardware deployment.

### Proposed end-to-end demonstration

```text
   PC A: DOOM / ViZDoom                 PC B: FPGA controller
┌─────────────────────────┐          ┌──────────────────────────────┐
│ Game engine and display │  frames  │ Observation / spike encoding │
│ State, reward, kills    │ ───────> │ Recurrent LIF spiking core   │
│ Executes received keys  │ <─────── │ Action decoding and output   │
└─────────────────────────┘ commands └──────────────────────────────┘
```

*Conceptual architecture only.* Frame acquisition, the PC-to-FPGA communication channel, action actuation, real-time timing, numerical quantization, board resource usage, and actual hardware-in-the-loop control remain to be designed, implemented, and measured. The diagram should not be interpreted as an already demonstrated system.

## Research questions

1. Can a visual agent learn to control ViZDoom from game frames rather than human keypresses?
2. What does a well-evaluated CNN baseline teach us about action choice, aiming, robustness, and experimental measurement?
3. Can a recurrent LIF spiking network learn a viable DOOM policy using reinforcement learning?
4. What is lost or preserved when an RSNN is quantized and mapped to FPGA arithmetic and memory resources?
5. Can the final hardware controller play DOOM on another computer with measurable gameplay performance and bounded latency?

## Research tracks and current status

| Track | Contribution | Status |
| --- | --- | --- |
| **CNN baseline** | DQN investigation, label-guided teacher, supervised RGB imitation CNN, explainability, brightness robustness | **Implemented and evaluated in software** |
| **RSNN development** | Recurrent LIF networks, spike encoding, RSNN v1 exploratory policy learning, RSNN v2 actor–critic training | **Software prototypes implemented; RSNN v2 training in progress** |
| **FPGA / two-PC integration** | Quantized neural inference on FPGA, frame transfer, action commands, hardware-in-the-loop validation | **Future implementation** |

The team uses **[sujeevanferos/projectDOOM](https://github.com/sujeevanferos/projectDOOM)** as the primary integration repository. The independently developed SNN research is maintained in **[DakshayaniRamanesh/project_DOOM](https://github.com/DakshayaniRamanesh/project_DOOM)** and linked here under **[`SNN-build/`](SNN-build/)** as a Git submodule. The CNN track is maintained here under **[`CNN-build/`](CNN-build/)**.

## CNN research: why it was built and what it established

The first reward-only CNN/Deep Q-Network baseline exhibited unstable and often ineffective policies in the reported experiments. We therefore developed a rule-based controller that uses ViZDoom's visible-object labels **only while producing demonstrations**. It generated 1,000 successful gameplay episodes comprising 5,781 RGB frame–action samples. A CNN trained by supervised **imitation learning** subsequently selected actions from RGB input **without accessing the labels buffer**.

The CNN observes RGB tensors of **3 × 60 × 80**, chooses among **left, right, attack**, and predicts action durations of **2, 4, or 8 tics**. The CNN v2 variant retains its architecture and adds brightness augmentation during training.

| CNN result | Recorded observation |
| --- | --- |
| Teacher data | 1,000 successful demonstrations; 5,781 frame–action decisions |
| CNN v1 held-out action agreement | 99.56% at epoch 50 |
| CNN v2 held-out action agreement | 99.74% at epoch 50 |
| CNN v1 dark-input test | 98/100 kills at 0.50× brightness; 97/100 at 0.75× |
| CNN v2 extended robustness verification | 3,000/3,000 observed kills: 1,000 episodes each at 1.0×, 0.50×, and 0.75× |

These results apply to the constrained **ViZDoom BASIC** scenario, not general DOOM. Supervised agreement with a teacher and gameplay kill success measure different things; 100% *observed* success does not imply perfect future reliability. Brightness was multiplied into the **CNN input**, not changed in the game environment. CNN v1/v2 tests used independent, unpaired episodes.

For methods, experiment numbering, limitations, and source CSVs, see **[CNN methodology](CNN-build/docs/METHODOLOGY.md)** and **[experiment record](CNN-build/docs/EXPERIMENTS.md)**. Experiment **5A** concerns brightness testing, **5B** feature visualization/Grad-CAM, and **5C** brightness-augmented training.

## RSNN development: current team approach

The SNN research track explores **recurrent spiking neural networks** whose neurons accumulate membrane potential, fire discrete spikes, and retain state across successive game decisions. The teammate's newer **RSNN v2** code introduces:

- **Input:** 32 × 24 grayscale pixels (768 values), converted to rate-coded spikes across a default eight neural timesteps per game decision.
- **Spiking core:** 96 leaky integrate-and-fire (LIF) units with learnable **96 × 96** recurrent synaptic feedback. Membrane and spike states persist within an episode and reset between episodes.
- **Learning:** on-policy, single-environment Advantage Actor–Critic (A2C) with generalized advantage estimation (**γ = 0.99, λ = 0.95**), a separate scalar critic, policy-entropy regularization, truncated backpropagation through time, and gradient clipping at norm **0.5**.
- **Surrogate gradients:** fast-sigmoid derivative `1 / (1 + 2|u − threshold|)^2`, allowing gradient-based optimization despite hard spikes during the forward pass.
- **Policy:** six outputs—noop, left, right, shoot, left+shoot, right+shoot—rather than the CNN's three actions and separate duration head.
- **Hardware-oriented accounting:** 73,728 input connection weights + 9,216 recurrent + 576 actor = **83,520 connection weights**; including input and actor biases yields **83,622 actor/core parameters**. The **97-parameter critic** is for training, leading to **83,719 total trainable parameters** in the default actor–critic model.

**Status qualification:** These architectural and algorithmic features are drawn from the teammate's current upstream RSNN v2 implementation and her supplied design notes. **RSNN v2 is still being trained; this README makes no claim that it has already learned a successful policy.** The resource numbers count model parameters, *not* synthesized FPGA utilization. See [RSNN architecture and hardware limitations](docs/RSNN_AND_HARDWARE.md).

**Submodule note:** The main repository pins a specific commit of the teammate's repository, and the pinned revision can lag newer upstream code. At the time of this documentation update, the pinned revision does not include the newer v2 files visible upstream. Updating `SNN-build` to an agreed teammate commit is a **separate integration step**; this documentation change does not alter its pointer or modify her repository.

### Why a direct CNN–RSNN performance comparison needs care

The models currently differ in input representation (RGB versus grayscale), spatial resolution, action vocabulary, temporal state, and learning method. We should not compare their validation accuracy or reward figures as if they arose from identical experiments. A later matched benchmark must standardize scenario settings, observation timing, success definition, episode budgets, and evaluation seeds.

## Repository layout

```text
projectDOOM/
├── README.md                   # Team overview and current progress
├── docs/
│   ├── RSNN_AND_HARDWARE.md    # RSNN v2 design, accounting and limitations
│   └── INTEGRATION_ROADMAP.md  # Work needed for fair benchmarks and FPGA control
├── CNN-build/
│   ├── cnn_research/           # Teacher, DQN, imitation, evaluation, explainability
│   ├── docs/                   # CNN methodology, results, migration, reproducibility
│   ├── tests/
│   └── runs/                   # Selected lightweight CSV/JSON evidence
├── SNN-build/                  # Git submodule: teammate's SNN codebase
├── .gitmodules
└── requirements.txt            # CNN-focused dependency inventory (not pinned)
```

## Getting started

Clone both research tracks (at the **submodule commit pinned by the main repository**):

```bash
git clone --recurse-submodules https://github.com/sujeevanferos/projectDOOM.git
cd projectDOOM
```

For an existing checkout: `git submodule update --init --recursive`. This does **not** fetch the latest teammate RSNN v2 training changes into the pinned main repository automatically.

For the reorganized CNN package, work from `CNN-build/`:

```bash
cd CNN-build
python -m cnn_research.teacher.labelled_aiming_agent_safe --help
python -m cnn_research.training.train_imitation_cnn --help
python -m cnn_research.training.train_imitation_cnn_brightness --help
python -m cnn_research.evaluation.verify_cnn_v2 --help
```

See [CNN reproducibility](CNN-build/docs/REPRODUCIBILITY.md) and [migration guide](CNN-build/docs/MIGRATION.md) for exact commands. Most training datasets (`.npz`) and checkpoints (`.pt`) are deliberately excluded from Git and must be regenerated or restored locally; the repository is not a complete pretrained-model distribution.

For the newer RSNN v2 scripts, use the teammate's **current upstream checkout** and [corrected-v2 guide](https://github.com/DakshayaniRamanesh/project_DOOM/blob/master/rsnn_doom_quickstart/README_CORRECTED_V2.md). Example *offline* checks after installing the relevant dependencies:

```bash
cd rsnn_doom_quickstart
python -m pytest -q tests/test_rsnn_v2_corrected.py
python train_rsnn_v2.py --mode smoke
```

A smoke test validates code behavior or differentiability, **not** a trained DOOM-playing policy or FPGA deployment. See the upstream guide before starting/resuming training.

## Roadmap to the actual goal

1. **Finish and validate RSNN v2 training:** report learning curves, kill rates, rewards, action frequencies, held-out seeds and failure analysis alongside random/fixed-action baselines.
2. **Establish a fair software benchmark:** align action protocols, conditions, success metrics and test budgets before attempting CNN–RSNN comparative claims.
3. **Prepare hardware inference:** select an FPGA target, define spike/state data layouts, arithmetic precision and weight transport, and test quantization against floating-point RSNN traces.
4. **Implement the two-computer loop:** obtain visual observations on the game PC, transmit inputs to hardware, receive decoded control outputs, and apply them to the game without manual per-action decisions.
5. **Measure end-to-end operation:** gameplay success, per-decision latency, interface throughput, timing closure, FPGA logic/memory/DSP utilization, power where measurable, and reliability across repeated sessions.

Details and stage-gate acceptance tests: [Integration roadmap](docs/INTEGRATION_ROADMAP.md).

## Research limitations and attribution

The project is ongoing, and this is a research repository rather than a peer-reviewed result. Source-based CNN and RSNN claims should be interpreted within the experiments actually run. A future FPGA implementation cannot be inferred from parameter counts alone. The CNN and RSNN repositories represent complementary contributions from team members; the upstream [RSNN repository](https://github.com/DakshayaniRamanesh/project_DOOM) retains its separate attribution, development history, and license considerations. Please reference the specific repository commits and experiment configurations when reproducing results.
