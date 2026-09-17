"""
Live weight-structure census for the online Q-network during training.

Tracks, per weight tensor, how far training has collapsed it away from its
random init: effective_dim (participation ratio of the eigenvalue spectrum
of W^T W, in the (in, out) convention -- PR = (Sigma lambda)^2 / Sigma
lambda^2; a single dominant direction -> PR near 1, uniform spread -> PR
near min(in, out)) and frobenius_drift (||W_now - W_init|| / ||W_init||).

No analytic null applies to these nets (biases + a trained embedding table
break the homogeneity assumptions a from-scratch analytic MP floor would
need), so this only ever compares against this run's OWN reconstructed
init -- exactly what you get for free by snapshotting state_dict() right
after construction, before any --resume-from load. That's still enough to
answer the practical question: is training collapsing a layer's usable
rank, and how far, and when.

Context (from a first offline pass, run 022): net.0 (input) collapsed from
eff_dim 471 -> 1.7 over 1.2M steps, net.2 (hidden) 256 -> 3.5, net.4
(readout) 275 -> 20.7, while the embedding table barely moved (11.9 ->
11.2) -- and critically, the two linear layers were STILL near-init at 10k
steps, only collapsing hard by 300k. That's a plausible live signal for
the rank-collapse / capacity-loss / plasticity-loss failure mode documented
in DRL (Kumar et al. 2020; Nikishin et al. "The Primacy Bias in Deep RL";
Lyle et al. on plasticity loss) -- distinct from supervised overfitting,
and worth watching rather than only measuring after the fact.
"""
from __future__ import annotations

import numpy as np
import torch


def participation_ratio(eigs: np.ndarray) -> float:
    eigs = np.maximum(eigs, 0.0)
    s = eigs.sum()
    return float((s ** 2) / np.sum(eigs ** 2)) if s > 0 else 0.0


def effective_dim(W: torch.Tensor) -> float:
    """W: a weight tensor in PyTorch's (out, in) convention (nn.Linear.weight
    or nn.Embedding.weight). Transposed to (in, out) so the eigendecomposition
    is always done on the smaller in x in or out x out gram matrix."""
    Wt = W.detach().float().cpu().numpy().T.astype(np.float64)
    n, d = Wt.shape
    gram = Wt.T @ Wt / max(n - 1, 1) if d <= n else Wt @ Wt.T / max(d - 1, 1)
    eigs = np.linalg.eigvalsh(gram)
    return participation_ratio(eigs)


def frobenius_drift(w_init: torch.Tensor, w_now: torch.Tensor) -> float:
    init_np = w_init.detach().float().cpu().numpy()
    now_np = w_now.detach().float().cpu().numpy()
    denom = np.linalg.norm(init_np)
    return float(np.linalg.norm(now_np - init_np) / denom) if denom > 0 else 0.0


def weight_keys(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Every weight matrix worth tracking: Linear and Embedding weights.
    Biases are excluded -- rank collapse is a statement about the weight
    matrix's column space, not a vector's magnitude."""
    return [k for k in state_dict if k.endswith(".weight")]


def structure_fields(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """CSV field list for MetricsLogger, derived from the actual network
    (handles both DQN and DecomposedDQN, and any hidden_layers count)."""
    fields = ["step"]
    for key in weight_keys(state_dict):
        fields += [f"eff_dim__{key}", f"drift__{key}"]
    return fields


def compute_structure_row(step: int, init_state: dict[str, torch.Tensor],
                          current_state: dict[str, torch.Tensor]) -> dict:
    row: dict = {"step": step}
    for key in weight_keys(init_state):
        row[f"eff_dim__{key}"] = round(effective_dim(current_state[key]), 2)
        row[f"drift__{key}"] = round(frobenius_drift(init_state[key], current_state[key]), 4)
    return row
