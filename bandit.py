"""Contextual bandits: off policy evaluation and online learners."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stimsafe.data.schema import CONTEXT, N_ARMS


def off_policy_estimates(policy_arm: np.ndarray, logged_arm: np.ndarray, reward: np.ndarray, propensity: np.ndarray,
                         predicted_reward: np.ndarray, rounds: int = 200, seed: int = 0) -> dict:
    """Estimate the value of a deterministic policy from logged data.

    Returns the direct method (model only), inverse propensity scoring, its
    self normalised form and the doubly robust estimator, each with a
    bootstrap 95 percent interval. Doubly robust stays consistent if either
    the reward model or the propensities are right.
    """
    n = len(reward)
    rows = np.arange(n)
    match = (policy_arm == logged_arm).astype(float)
    weight = match / np.clip(propensity, 1e-3, None)
    dm_terms = predicted_reward[rows, policy_arm]
    ips_terms = weight * reward
    dr_terms = dm_terms + weight * (reward - predicted_reward[rows, logged_arm])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, (rounds, n))

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    snips_boot = (ips_terms[index].sum(axis=1)) / np.clip(weight[index].sum(axis=1), 1e-9, None)
    return {
        "direct_method": float(dm_terms.mean()), "direct_method_ci": interval(dm_terms[index].mean(axis=1)),
        "ips": float(ips_terms.mean()), "ips_ci": interval(ips_terms[index].mean(axis=1)),
        "snips": float(ips_terms.sum() / max(weight.sum(), 1e-9)), "snips_ci": interval(snips_boot),
        "doubly_robust": float(dr_terms.mean()), "doubly_robust_ci": interval(dr_terms[index].mean(axis=1)),
        "match_rate": float(match.mean()), "effective_sample_size": float(weight.sum() ** 2 / max((weight ** 2).sum(), 1e-9)),
    }


def bandit_features(context: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Standardised context with a bias term and simple non linear terms, for the linear bandits."""
    X = context[CONTEXT].astype(float).copy()
    X["previous_oocytes"] = X["previous_oocytes"].fillna(0)
    X["log_amh"] = np.log(X["amh_ng_ml"])
    X["log_afc"] = np.log1p(X["afc"])
    values = X.to_numpy()
    mean = values.mean(axis=0) if mean is None else mean
    std = values.std(axis=0) + 1e-9 if std is None else std
    return np.column_stack([np.ones(len(X)), (values - mean) / std]), mean, std


class LinUCB:
    """Disjoint LinUCB: one ridge regression per arm with an upper confidence bonus."""

    def __init__(self, n_features: int, n_arms: int = N_ARMS, alpha: float = 0.6):
        self.alpha = alpha
        self.A_inv = np.stack([np.eye(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))

    def scores(self, x: np.ndarray) -> np.ndarray:
        theta = np.einsum("aij,aj->ai", self.A_inv, self.b)
        bonus = np.sqrt(np.einsum("i,aij,j->a", x, self.A_inv, x))
        return theta @ x + self.alpha * bonus

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        Ax = self.A_inv[arm] @ x
        self.A_inv[arm] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)      # Sherman Morrison rank one update
        self.b[arm] += reward * x


def run_online_simulation(features: np.ndarray, sample_reward, true_values: np.ndarray, clinician_probs: np.ndarray,
                          allowed: np.ndarray | None, alpha: float, epsilon: float, seed: int = 0) -> dict[str, np.ndarray]:
    """Play LinUCB, epsilon greedy and the clinician heuristic through the simulator.

    ``sample_reward(i, arm)`` returns a realised reward for patient ``i``.
    Regret is measured against the best arm by true expected reward. When
    ``allowed`` is given, a safe LinUCB variant may only pull arms that pass
    the safety filter, which shows the cost of learning under constraints.
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape
    learners = {"LinUCB": LinUCB(d, alpha=alpha), "Epsilon greedy": LinUCB(d, alpha=0.0)}
    if allowed is not None:
        learners["LinUCB with safety filter"] = LinUCB(d, alpha=alpha)
    best = true_values.max(axis=1)
    regret = {name: np.zeros(n) for name in [*learners, "Clinician heuristic"]}
    for i in range(n):
        x = features[i]
        for name, learner in learners.items():
            scores = learner.scores(x)
            if name == "Epsilon greedy" and rng.random() < epsilon:
                arm = int(rng.integers(0, N_ARMS))
            elif name == "LinUCB with safety filter":
                arm = int(np.where(allowed[i], scores, -np.inf).argmax())
            else:
                arm = int(scores.argmax())
            learner.update(arm, x, sample_reward(i, arm))
            regret[name][i] = best[i] - true_values[i, arm]
        clinician_arm = int(rng.choice(N_ARMS, p=clinician_probs[i]))
        regret["Clinician heuristic"][i] = best[i] - true_values[i, clinician_arm]
    return {name: np.cumsum(values) for name, values in regret.items()}
