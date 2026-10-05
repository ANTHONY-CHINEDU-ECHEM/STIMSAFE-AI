"""Serving facade used by the API and the dashboard."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from stimsafe.config import load_config, resolve
from stimsafe.data.schema import CONTEXT, CONTEXT_RANGES, DOSES_IU, N_REGIMENS, REGIMENS, describe_arm
from stimsafe.data.simulate import clinician_policy
from stimsafe.policy.safety import allowed_arms, risk_reasons


class InvalidPatientError(ValueError):
    """Raised when a patient record is outside plausible clinical ranges."""


def normalise_patient(patient: dict) -> dict:
    """Validate ranges and apply first cycle logic."""
    clean = {}
    for name in CONTEXT:
        value = patient.get(name)
        if name == "previous_oocytes" and (value is None or patient.get("first_cycle", 1)):
            clean[name] = np.nan
            continue
        if value is None:
            if name in {"pcos", "previous_ohss"}:
                value = 0
            elif name == "first_cycle":
                value = 1
            else:
                raise InvalidPatientError(f"{name} is required.")
        low, high = CONTEXT_RANGES[name]
        if not low <= float(value) <= high:
            raise InvalidPatientError(f"{name} must be between {low} and {high}.")
        clean[name] = float(value)
    if clean["first_cycle"] == 1:
        clean["previous_ohss"] = 0.0
    return clean


class ProtocolRecommender:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.models, self.classifier, self.safety = bundle["models"], bundle["classifier"], bundle["safety"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProtocolRecommender":
        path = Path(path) if path else resolve(load_config().artifacts.bundle)
        if not path.exists():
            raise FileNotFoundError(f"No model bundle at {path}. Run `make all` to train one.")
        return cls(joblib.load(path))

    def recommend(self, patient: dict) -> dict:
        """Recommended protocol, the full option table and the safety assessment for one patient."""
        clean = normalise_patient(patient)
        context = pd.DataFrame([clean])[CONTEXT]
        predicted = {k: v[0] for k, v in self.models.predict_all_arms(context).items()}
        reasons = risk_reasons({k: (None if pd.isna(v) else v) for k, v in clean.items()}, self.safety)
        rule_allowed = allowed_arms(context, self.safety)[0]
        allowed = allowed_arms(context, self.safety, predicted["ohss"][None, :])[0]
        best = int(np.where(allowed, predicted["reward"], -np.inf).argmax())
        unconstrained = int(predicted["reward"].argmax())

        def option(arm: int) -> dict:
            blocked = None
            if not rule_allowed[arm]:
                blocked = "Outside the high risk protocol restrictions"
            elif not allowed[arm]:
                blocked = f"Predicted OHSS risk above {100 * self.safety['max_predicted_ohss_risk']:.0f} percent"
            return {**describe_arm(arm), "expected_oocytes": round(float(predicted["oocytes"][arm]), 1),
                    "oocytes_low": round(float(predicted["oocytes_low"][arm]), 1), "oocytes_high": round(float(predicted["oocytes_high"][arm]), 1),
                    "ohss_risk": round(float(predicted["ohss"][arm]), 4), "expected_utility": round(float(predicted["reward"][arm]), 4),
                    "allowed": bool(allowed[arm]), "blocked_reason": blocked}

        clinician = int(clinician_policy(context)[0].argmax())
        constraints = []
        if reasons:
            constraints = [f"Starting dose capped at {self.safety['max_dose_when_flagged_iu']} IU",
                           "Antagonist protocol with GnRH agonist trigger only"]
        return {
            "recommendation": option(best),
            "safety": {"high_risk": bool(reasons), "reasons": reasons, "constraints": constraints,
                       "filter_changed_recommendation": best != unconstrained,
                       "above_risk_ceiling": bool(predicted["ohss"][best] > self.safety["max_predicted_ohss_risk"]),
                       "unconstrained_choice": describe_arm(unconstrained)["label"] if best != unconstrained else None},
            "options": [option(a) for a in range(len(predicted["reward"]))],
            "comparators": {"usual_practice": option(clinician), "classifier": option(int(self.classifier.predict_arm(context)[0]))},
            "model_version": self.bundle["version"],
        }

    def similar_patients(self, patient: dict) -> dict:
        """Historical outcomes by starting dose among the most similar past patients."""
        clean = normalise_patient(patient)
        store = self.bundle["similar"]
        query = np.array([clean["age"], np.log(clean["amh_ng_ml"]), clean["afc"], clean["bmi"], clean["pcos"]])
        k = min(self.bundle["neighbours"], len(store["outcomes"]))
        _, index = store["index"].kneighbors(((query - store["mean"]) / store["std"] * store["weights"])[None, :], n_neighbors=k)
        cohort = store["outcomes"].iloc[index[0]]
        groups = []
        for dose in DOSES_IU:
            part = cohort[cohort["dose_iu"] == dose]
            groups.append({"dose_iu": dose, "cycles": int(len(part)),
                           "mean_oocytes": round(float(part["oocytes"].mean()), 1) if len(part) else None,
                           "optimal_response_rate": round(float(part["oocytes"].between(8, 18).mean()), 3) if len(part) else None,
                           "poor_response_rate": round(float((part["oocytes"] < 4).mean()), 3) if len(part) else None,
                           "ohss_rate": round(float(part["ohss"].mean()), 3) if len(part) else None})
        return {"cohort_size": int(len(cohort)), "by_dose": groups}

    def policy_table(self) -> list[dict]:
        return [{"policy": name, **{k: v for k, v in row.items() if k != "off_policy"},
                 "doubly_robust_estimate": row["off_policy"].get("doubly_robust")} for name, row in self.bundle["policy_table"].items()]


@lru_cache(maxsize=1)
def get_recommender() -> ProtocolRecommender:
    return ProtocolRecommender.load()


__all__ = ["ProtocolRecommender", "InvalidPatientError", "get_recommender", "normalise_patient", "REGIMENS", "N_REGIMENS"]
