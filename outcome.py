"""Outcome models: what happens to this patient on each candidate protocol?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier

from stimsafe.data.schema import CONTEXT, DOSES_IU, N_ARMS, N_REGIMENS

ACTION_FEATURES = ["dose_iu", "regimen_antagonist_agonist", "regimen_long_agonist"]
FEATURES = CONTEXT + ACTION_FEATURES


def design_matrix(context: pd.DataFrame, arm: np.ndarray) -> pd.DataFrame:
    """Join patient context with an encoded action."""
    arm = np.asarray(arm)
    X = context[CONTEXT].reset_index(drop=True).astype(float).copy()
    X["dose_iu"] = np.asarray(DOSES_IU, dtype=float)[arm // N_REGIMENS]
    X["regimen_antagonist_agonist"] = (arm % N_REGIMENS == 1).astype(float)
    X["regimen_long_agonist"] = (arm % N_REGIMENS == 2).astype(float)
    return X


class OutcomeModels:
    """Gradient boosted models of reward, oocyte yield (with an 80 percent band) and OHSS risk.

    Monotonic constraints encode pharmacology the data should never be allowed
    to contradict: a higher dose cannot lower expected yield or OHSS risk.
    """

    def __init__(self, seed: int = 0):
        dose_up = [1 if f == "dose_iu" else 0 for f in FEATURES]
        common = {"learning_rate": 0.06, "max_iter": 300, "max_leaf_nodes": 15, "l2_regularization": 1.0, "random_state": seed}
        self.reward = HistGradientBoostingRegressor(**common)
        self.oocytes = HistGradientBoostingRegressor(loss="poisson", monotonic_cst=dose_up, **common)
        self.oocytes_low = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, monotonic_cst=dose_up, **common)
        self.oocytes_high = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, monotonic_cst=dose_up, **common)
        self.ohss = HistGradientBoostingClassifier(monotonic_cst=dose_up, **common)

    def fit(self, log: pd.DataFrame) -> "OutcomeModels":
        X = design_matrix(log, log["arm"].to_numpy())
        self.reward.fit(X, log["reward"])
        self.oocytes.fit(X, log["oocytes"])
        self.oocytes_low.fit(X, log["oocytes"])
        self.oocytes_high.fit(X, log["oocytes"])
        self.ohss.fit(X, log["ohss"])
        return self

    def predict_all_arms(self, context: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predictions for every patient and arm, each an array of shape (patients, arms)."""
        n = len(context)
        stacked = pd.concat([context] * N_ARMS, ignore_index=True)
        X = design_matrix(stacked, np.repeat(np.arange(N_ARMS), n))
        shape = lambda values: np.asarray(values).reshape(N_ARMS, n).T  # noqa: E731
        return {"reward": shape(self.reward.predict(X)), "oocytes": shape(self.oocytes.predict(X)),
                "oocytes_low": shape(self.oocytes_low.predict(X)), "oocytes_high": shape(self.oocytes_high.predict(X)),
                "ohss": shape(self.ohss.predict_proba(X)[:, 1])}


def successful_cycles(log: pd.DataFrame) -> pd.Series:
    """Cycles with the outcome every clinician wants: a good yield with no OHSS."""
    return (log["oocytes"].between(8, 18)) & (log["ohss"] == 0)


class ProtocolClassifier:
    """Multi output classifier that imitates the protocols used in successful historical cycles.

    One estimator per decision: starting dose category, protocol and trigger.
    """

    TARGETS = ["dose_index", "protocol_index", "trigger_index"]

    def __init__(self, seed: int = 0):
        base = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=200, max_leaf_nodes=15, random_state=seed)
        self.model = MultiOutputClassifier(base)

    @staticmethod
    def targets(log: pd.DataFrame) -> np.ndarray:
        arm = log["arm"].to_numpy()
        regimen = arm % N_REGIMENS
        return np.column_stack([arm // N_REGIMENS, (regimen == 2).astype(int), (regimen == 1).astype(int)])

    def fit(self, log: pd.DataFrame) -> "ProtocolClassifier":
        good = log[successful_cycles(log)]
        self.n_training_cycles_ = len(good)
        self.model.fit(good[CONTEXT].astype(float), self.targets(good))
        return self

    def predict_arm(self, context: pd.DataFrame) -> np.ndarray:
        dose, long_agonist, agonist_trigger = self.model.predict(context[CONTEXT].astype(float)).T
        regimen = np.where(long_agonist == 1, 2, np.where(agonist_trigger == 1, 1, 0))
        return dose * N_REGIMENS + regimen
