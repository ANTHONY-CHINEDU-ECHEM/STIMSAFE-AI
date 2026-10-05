"""Rule based OHSS safety filter.

The learned policy is never allowed the last word on safety. Before any
recommendation is shown, the patient is screened against explicit clinical
thresholds. A flagged patient has the dose capped and is restricted to an
antagonist protocol with a GnRH agonist trigger, the combination that most
reduces the risk of ovarian hyperstimulation syndrome. Thresholds live in the
configuration file so a clinic can align them with its own guideline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stimsafe.data.schema import DOSES_IU, N_ARMS, N_REGIMENS

SAFE_REGIMEN = 1   # antagonist protocol with GnRH agonist trigger


def risk_reasons(patient: dict, cfg: dict) -> list[str]:
    """Plain language reasons a patient is treated as high risk for OHSS."""
    reasons = []
    if patient.get("pcos"):
        reasons.append("Polycystic ovary syndrome")
    if patient.get("amh_ng_ml") is not None and patient["amh_ng_ml"] >= cfg["amh_high_ng_ml"]:
        reasons.append(f"AMH of {patient['amh_ng_ml']:.1f} ng/mL is at or above {cfg['amh_high_ng_ml']}")
    if patient.get("afc") is not None and patient["afc"] >= cfg["afc_high"]:
        reasons.append(f"Antral follicle count of {patient['afc']:.0f} is at or above {cfg['afc_high']}")
    if patient.get("previous_ohss"):
        reasons.append("OHSS in a previous cycle")
    previous = patient.get("previous_oocytes")
    if previous is not None and not pd.isna(previous) and previous >= cfg["previous_high_response_oocytes"]:
        reasons.append(f"Previous cycle yielded {previous:.0f} oocytes")
    if patient.get("age", 99) < cfg["young_age"] and patient.get("bmi", 99) < cfg["low_bmi"]:
        reasons.append(f"Younger than {cfg['young_age']} with BMI under {cfg['low_bmi']}")
    return reasons


def high_risk_mask(context: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Vectorised version of :func:`risk_reasons` for whole cohorts."""
    previous = context["previous_oocytes"].fillna(0)
    return ((context["pcos"] == 1) | (context["amh_ng_ml"] >= cfg["amh_high_ng_ml"]) | (context["afc"] >= cfg["afc_high"])
            | (context["previous_ohss"] == 1) | (previous >= cfg["previous_high_response_oocytes"])
            | ((context["age"] < cfg["young_age"]) & (context["bmi"] < cfg["low_bmi"]))).to_numpy()


def allowed_arms(context: pd.DataFrame, cfg: dict, predicted_ohss: np.ndarray | None = None) -> np.ndarray:
    """Boolean matrix (patients by arms) of actions that pass the safety filter.

    Two layers apply. Flagged patients are limited to capped doses on the safe
    regimen. Then, for every patient, any arm whose predicted OHSS risk
    exceeds the configured ceiling is removed. If the second layer would
    remove everything, the arm with the lowest predicted risk is kept.
    """
    n = len(context)
    allowed = np.ones((n, N_ARMS), dtype=bool)
    flagged = high_risk_mask(context, cfg)
    dose_of_arm = np.repeat(DOSES_IU, N_REGIMENS)
    regimen_of_arm = np.tile(np.arange(N_REGIMENS), len(DOSES_IU))
    safe = (dose_of_arm <= cfg["max_dose_when_flagged_iu"]) & (regimen_of_arm == SAFE_REGIMEN)
    allowed[flagged] = safe
    if predicted_ohss is not None:
        within = allowed & (predicted_ohss <= cfg["max_predicted_ohss_risk"])
        empty = ~within.any(axis=1)
        if empty.any():
            fallback = np.where(allowed[empty], predicted_ohss[empty], np.inf).argmin(axis=1)
            within[np.where(empty)[0], fallback] = True
        allowed = within
    return allowed
