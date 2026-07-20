# gated_node_lab

A reduced-order surrogate pipeline for forced structural dynamics:
sparse-sensor autoencoder + neural ODE, with dynamics blended by a
mixture of experts whose gate is driven by the **forcing input and its
causal history** (`rbf_moe`). No wall-clock gating, no assumption about
the forcing waveform shape.

## The model (`rbf_moe`)

`RBFHistoryMoE_NODE`: K expert MLPs on `[z, u]`, blended by an `RBFGate`
over causal forcing-history features `[h, u_gate]`:

- **History features.** The latent state is augmented to `[z | h]`,
  where `h` is a bank of leaky integrators `ḣ = (W_u u − h)/τ` with
  learnable per-feature timescales (log-spaced init from ~4·dt to
  t_max). This gives the gate causal memory of the forcing — the
  instantaneous value of `u` alone is ambiguous (same force level on the
  way up vs. down vs. between pulses).
- **Periodic history init.** Every training window carries its aligned
  forcing cycle (`U_period`); the unique periodic history state at the
  window's start phase is computed analytically inside the graph
  (`LeakyHistoryBank.periodic_initial_state`) — no from-rest transient,
  differentiable w.r.t. the history-bank parameters. The mechanical
  encoder predicts only `z`, never `h`.
- **Gate stability.** RBF sigma floor, temperature scheduling
  (`GateLabCallback`), gate-visible control channels
  (`gate_control_indices` — hide constant parameter channels so the gate
  cannot become a trajectory classifier), optional periodic phase
  features, and an optional non-uniform expert-usage prior
  (`gate_balance_target`) with regime anchor/overlap losses.
- **Data-driven seeding.** `init_rbf_gate_from_data` (k-means over
  rolled-out gate features) or `init_rbf_gate_from_segments` (supervised:
  expert k owns the k-th time window, e.g. pre-pulse / pulse /
  post-pulse), both of which then place the whole gating pathway on a
  reduced learning rate (`gate_lr_mult`).

## Layout

```
src/
  history.py       LeakyHistoryBank + analytic periodic initial state
  gates.py         RBFGate (+ torch-only k-means for seeding)
  gated_nodes.py   RBFHistoryMoE_NODE + GateProbe            (torch-only)
  EDM.py           base EDM class + shared graph blocks (encoder/decoder,
                   full-space lift, noise, checkpointed RK4, resampling)
  EDM_v1_1.py      + CST mesh-gradient spatial loss for 2-D vector_nodes
  EDM_v1_2.py      the rbf_moe EDM: graph wiring, gate losses, periodic
                   data API, seeding, training loop  (needs neuromancer)
Utils/
  beam_problem.py / beam_fea_v2_pointload_shared_mesh.py   FEA data gen
  sensor_processing_v2.py, clustering.py, data_processing.py, sobol.py,
  yaml_processor.py, psd.py, upsampler.py, trainer.py
Examples/
  omega/               1-D beam, forcing-frequency sweep
  tau_and_s/           1-D beam, pulse-shape (τ, s) sweep
  elliptic_hole_omega/ 2-D FEA beam with elliptic hole (vector_nodes + CST)
    each: Step_00_Create_Data → Step_01_Clustering →
          Step_02_Training (build_and_train.ipynb, method="rbf_moe") →
          Step_03_Evaluation (evaluate_n_plot_with_field_plots.ipynb)
tests/
  smoke_test.py        torch-only: shapes, RK4 rollout, grads, gate simplex,
                       periodic-init consistency
  integration_test.py  full Problem build + loss + backward + gate seeding
```

## Usage

```python
import sys
sys.path.insert(0, "/path/to/gated_node_lab/src")
sys.path.insert(0, "/path/to/gated_node_lab/Utils")

from EDM_v1_2 import EDM
model = EDM(A_Mat, pinv_Theta, dt, t_max, config_global, device)
model.build_model(method="rbf_moe")
model.init_rbf_gate_from_segments(ft_train, boundaries=(0.45, 0.55))  # optional
model.train_model(output_paths, ...)
```

## Config knobs (all optional, defaults in `EDM_v1_2.NODE_DEFAULTS`)

```yaml
model:
  n_experts: 3               # keep 2-3; more experts = harder to balance
  n_hist: 8                  # history features
  gate_temperature: 3.0      # initial T
  gate_t_min: 1.0            # annealing floor
  gate_anneal_gamma: 1.0     # per-epoch decay (1.0 = no annealing)
  gate_sigma_min: 0.10       # hard floor on RBF sigmas
  gate_lr_mult: 0.1          # gating-pathway LR multiplier after seeding
  gate_control_indices: [0]  # instantaneous controls visible to the gate
  gate_balance_target:       # optional non-uniform usage prior, e.g. [0.45, 0.10, 0.45]
  gate_regime_boundaries: [0.47, 0.53]   # anchor windows for GATEANCHOR/GATEOVERLAP
training:
  Qs:
    GATEBALANCE: 0       # expert-usage prior weight
    GATEANCHOR: 0        # regime anchor loss
    GATEOVERLAP: 0       # regime overlap loss
    EXPERTDIVERSITY: 0   # expert-output diversity loss
```

## Diagnostics

- Training prints `[GateLab] epoch .. T=.. usage=a/b/c` — batch-mean
  expert usage from the `GateProbe`. Healthy: no component pinned at 0.
- The evaluation notebooks plot per-expert gate schedules over the period
  and per-regime phase-plane flow fields (streamlines of the learned
  vector field around the trajectory, with pass-ambiguity masking and a
  data-calibrated kinematic horizontal component).

## Tests

```bash
python tests/smoke_test.py         # torch only
python tests/integration_test.py  # full graph (needs neuromancer)
```

## Note on old checkpoints

State dicts (`best_model_state_dict.pth`) trained before the refactor
load unchanged — the graph structure is identical. Full-model pickles
(`best_model.pth`) reference the old module name
(`EDM_v1_2_three_expert_unique`) and will not unpickle; use the state
dict, as the notebooks do.
