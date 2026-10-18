# projectDOOM: Integration and Validation Roadmap

**Primary objective:** demonstrate a hardware-hosted spiking controller operating DOOM on a *separate* PC, with defensible experiment records.

This roadmap distinguishes **completed CNN research**, **ongoing RSNN training**, and **future FPGA/two-PC implementation**. Stage transitions should rely on recorded measurements, not merely source-code existence.

## Stage 0 — Establish an empirical CNN benchmark (completed in constrained setting)

The team built a CNN baseline to learn practical mechanisms of autonomous visual gameplay. Completed experiments include a reward-only DQN baseline, a privileged-label aiming teacher, RGB imitation CNN v1, feature visualization, brightness sensitivity analysis, brightness-augmented CNN v2 and 3,000-episode v2 robustness verification.

The original CNN uses RGB 3 × 60 × 80, three single-button actions and learned 2/4/8-tic decision durations. Its highest reported gameplay success is in the **ViZDoom BASIC** scenario. These metrics establish the current *software reference*, not a universal performance target for other levels or hardware.

**Artifacts:** `CNN-build/docs/METHODOLOGY.md`, `CNN-build/docs/EXPERIMENTS.md`, `CNN-build/runs/**/summary.csv`.

## Stage 1 — Complete RSNN v2 learning and evaluation (ongoing)

The teammate's RSNN v2 uses grayscale 32 × 24, spike encoding, 96 recurrent LIF neurons and on-policy A2C/GAE training. It has a six-action readout. The RSNN v2 software exists upstream; its training is in progress.

**Suggested completion evidence:**
- Source commit and software/hardware/seed manifest for the *exact* trained RSNN v2 checkpoint.
- Repeated training runs or seeds where feasible, with reward/kill-rate learning curves.
- Policy action-frequency plots and checks against degenerate always-shoot behavior.
- Held-out evaluation seeds, separated from training and model selection.
- Comparisons against random and fixed-action baselines using the *same* RSNN environment configuration.
- Checkpoint SHA-256, test outcomes and confidence intervals on reported kill rates.

**Gate:** Do not claim a trained successful SNN controller merely because gradients flow or ViZDoom launches.

## Stage 2 — Fair CNN vs RSNN research benchmark (planned)

The existing models are deliberately *not identical*: RGB/gray input, resolution, recurrent state, action count, action timing, algorithm and frame control differ. First record these differences and decide which factors to standardize. Consider separately comparing (a) each model in its native configuration and (b) a controlled protocol aligned to the same input/action/timing interface.

Track episode reward, kill success, episode tics, number of decisions, model size, computational cost, visual perturbations and decision latency. Use disjoint evaluation seeds and specify whether episodes are paired. Report uncertainty; do not reduce the study to one accuracy number.

**Gate:** Published comparisons must state what is matched and what remains different.

## Stage 3 — RSNN inference interface specification (planned)

Design a stable, testable interface between a visual frame producer and a spiking controller. Define packet structures, message framing, clocking, action encoding, state-reset semantics, failure recovery and timing requirements.

A first offline reference should replay prerecorded frames into the software RSNN and record membrane/spike traces, logits and resulting actions. This becomes a numerical oracle for fixed-point/hardware testing.

**Gate:** Complete interface document plus repeatable software trace comparison.

## Stage 4 — FPGA mapping and validation (planned)

Select an FPGA board and toolchain; freeze one inference architecture. Quantize weights, thresholds and decay constants, and study fixed-point saturation and rounding. Determine memory representation, neuron execution scheduling and output readout.

**Required measurement:** synthesized LUT/register/DSP/BRAM use, achieved clock, worst-case per-decision latency, throughput, power where measurable and mismatch against a floating-point reference. The network's 83,622 actor/core trainable parameters alone do not determine these quantities.

**Gate:** Verified hardware inference on test vectors before connection to game controls.

## Stage 5 — Hardware-in-the-loop two-computer DOOM control (planned)

Connect **the game PC** to **the FPGA controller** with a defined communication channel. The game PC captures frames, transmits observations, receives action predictions, and applies the requested keypresses without per-decision human intervention.

Collect aligned timestamps for frame capture, transport, spike processing, action return and action execution. Ensure valid episode reset propagates to neuron state.

**Gate:** Repeatable gameplay experiments demonstrating decisions really originate from FPGA inference—not a host-side neural model—plus verified latency, failure recovery, and experiment logs.

## Stage 6 — Final report and reproducible release (planned)

Produce an academic project report with research questions, methods, hypotheses, baselines, learning curves, confidence intervals, ablations, hardware resource utilization and honest limitations. Pin both repositories to exact commits and archive large checkpoints/datasets outside ordinary Git history. Clearly credit each teammate's contribution.

## Repository collaboration

- Main integration repository: [sujeevanferos/projectDOOM](https://github.com/sujeevanferos/projectDOOM).
- Independent RSNN development: [DakshayaniRamanesh/project_DOOM](https://github.com/DakshayaniRamanesh/project_DOOM), linked by the `SNN-build` submodule.
- The main repository **pins** an RSNN commit; upstream pushes are not automatically integrated. Agree on the tested upstream revision, then update the submodule pointer in a separate pull request.
- Keep code changes, model training, documentation, and hardware integration in reviewable branches/PRs.
- The main README describes overall team goals; CNN methods/results reside under `CNN-build/docs/`, while RSNN methods and constraints are documented here and in the upstream repository.
