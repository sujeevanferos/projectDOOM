# Recurrent Spiking Neural Network (RSNN) Architecture and Hardware Considerations

**Track:** team SNN research. **Status:** RSNN v2 software implementation and ongoing training; FPGA implementation unverified.

## Scientific purpose

The recurrent SNN is the intended neural controller for the project's eventual **external FPGA-to-DOOM** experiment. The CNN pipeline supplies a working vision-control baseline and evaluation lessons, while the RSNN explores dynamical spike-based representations that might be mapped to a neuromorphic processor or FPGA implementation. Successful software simulation is necessary but insufficient for validating real-time hardware control.

## Provenance and version boundary

This description is based on (1) the team's supplied RSNN v2 architecture/training notes and (2) the independently maintained [teammate repository](https://github.com/DakshayaniRamanesh/project_DOOM), particularly `rsnn_doom_quickstart/rsnn_v2.py`, `doom_env.py`, `train_rsnn_v2.py`, and `README_CORRECTED_V2.md`. The main project's `SNN-build` **Git submodule is pinned to an older upstream commit** that may not contain these newer files. The upstream RSNN v2 is currently **being trained**; documentation must not extrapolate a kill rate, learning curve, successful FPGA run or finished training from the code alone.

The older RSNN v1 experiment used an exploratory REINFORCE approach. The updated RSNN v2 changes the software learner to actor–critic with GAE and an explicit critic head.

## Signal pathway

```text
ViZDoom game frame
   ↓ grayscale conversion + resize
32 × 24 grayscale observation → 768 normalized intensity values
   ↓ deterministic rate spike encoding, 8 simulated neural steps
96 recurrent LIF neurons, including learned recurrent spike feedback
   ├── 6-action actor head → game-control decision
   └── scalar critic V(s) → training loss only
```

The six actor actions in the present RSNN code are `noop`, `left`, `right`, `shoot`, `left_shoot`, `right_shoot`. They differ from the CNN's three single-button actions and separately learned action duration. This is a major benchmarking confound that must be addressed before claiming comparative superiority.

## LIF dynamics and gradient estimation

The hidden layer has **96 leaky integrate-and-fire neurons** with learnable feedforward and recurrent weights. Its configuration includes membrane decay `beta=0.5` by default (allowed values 0.25, 0.5, 0.75) and threshold 0.4. The model preserves membrane and previous spike state between successive game decisions **within the same episode**, and resets both states when starting a new episode. Rollout boundaries for truncated BPTT detach computational history without restarting the episode or wiping neural state.

A hard threshold is applied in the forward pass, while the backward pass uses a fast-sigmoid surrogate gradient:

```math
\frac{\partial s}{\partial u} \approx \frac{1}{\big(1 + 2\lvert u - V_{\mathrm{th}}\rvert\big)^2}.
```

This is a software gradient estimator; it is not evidence that real FPGA neurons have differentiable spikes.

## RSNN v2 software learning algorithm

The updated implementation employs single-environment on-policy **Advantage Actor–Critic (A2C)** with generalized advantage estimation (GAE). Defaults in the current code include discount factor **gamma = 0.99**, GAE trace **lambda = 0.95**, entropy coefficient **0.01**, critic loss coefficient **0.5**, and gradient clipping at maximum norm **0.5**.

The training program forms short recurrent rollout segments, computes returns and advantages with appropriate bootstrapping when a next observation is available, and performs truncated backpropagation through time. Policy-entropy regularization discourages early deterministic action collapse. The model's **critic head is for training**; the intended hardware inference path retains the spiking core and actor head.

The corrected-v2 upstream notes describe a particular testing and checkpointing protocol, including per-policy evaluation isolation and separate validation seeds. Their offline tests/smoke tests are *software verification*, not gameplay performance evidence. RSNN v2 training is still running, and no final performance figure is included here.

## Network and resource accounting

The implemented PyTorch model declares these default tensor dimensions:

| Component | Connections / parameters |
| --- | ---: |
| Feedforward weights: 768 → 96 | 73,728 |
| Recurrent weights: 96 → 96 | 9,216 |
| Actor weights: 96 → 6 | 576 |
| **Actor/core connection weights** | **83,520** |
| Input layer bias | 96 |
| Actor bias | 6 |
| **Actor/core weights and biases** | **83,622** |
| Critic weights and bias | 97 |
| **Total trainable RSNN v2 parameters** | **83,719** |
| LIF hidden neurons | **96** |

The initial team notes mentioned **83,616** as an actor/core count; the *current implementation's* tensor dimensions imply **83,520 connection weights**, or **83,622 weights plus biases**. This document uses the code-derived count and does not alter the implementation. Verify resource summaries again when the model architecture changes.

**These counts are not FPGA resource utilization.** They do not establish how many on-chip memory bits, DSP slices, LUTs, registers, clocks, synapse updates or spike packets a specific board needs. Recurrent connectivity and state retention can impose substantial memory bandwidth and timing constraints.

## FPGA design questions to resolve

1. **Target hardware:** selected FPGA board, device family, available BRAM/URAM, LUTs, DSPs, clocks, I/O and programming/toolchain constraints.
2. **Numeric representation:** fixed-point width, scale, rounding, saturation, membrane leak and threshold accuracy; quantify deviation against floating-point traces.
3. **Neural execution:** mapping of 768 inputs, 96 LIF states, 9,216 recurrent weights, synaptic accumulation and eight internal timesteps onto serial/parallel logic.
4. **Communication:** image-frame acquisition, PC-to-board transport protocol, data integrity/sequence IDs, action-return channel and recovery after lost messages.
5. **Control loop:** measure latency from acquired frame to executed game action, controller update rate, and whether action commands remain valid while the game advances.
6. **Evaluation:** compare software RSNN and hardware RSNN on recorded observation sequences, then run actual closed-loop gameplay from the external board.

**Current limitation:** No confirmed fixed-point conversion, synthesis, timing closure, actual FPGA power measurements or two-computer gameplay trial is reported by the available material.

## Sources within this project

- [Team-level README](../README.md)
- [Integration roadmap](INTEGRATION_ROADMAP.md)
- [CNN methodology](../CNN-build/docs/METHODOLOGY.md)
- [Upstream RSNN v2 implementation](https://github.com/DakshayaniRamanesh/project_DOOM/blob/master/rsnn_doom_quickstart/rsnn_v2.py)
- [Upstream corrected RSNN v2 instructions](https://github.com/DakshayaniRamanesh/project_DOOM/blob/master/rsnn_doom_quickstart/README_CORRECTED_V2.md)

The upstream repository evolves independently. Pin a specific commit before publishing formal experiment claims.
