"""Action space and patient schema shared across the project."""
from __future__ import annotations

import numpy as np

DOSES_IU = [100, 150, 225, 300]                       # daily gonadotropin starting dose categories
REGIMENS = [                                          # protocol and trigger combinations that are clinically valid
    {"key": "antagonist_hcg", "protocol": "antagonist", "trigger": "hcg", "label": "Antagonist protocol, hCG trigger"},
    {"key": "antagonist_agonist", "protocol": "antagonist", "trigger": "gnrh_agonist", "label": "Antagonist protocol, GnRH agonist trigger"},
    {"key": "long_agonist_hcg", "protocol": "long_agonist", "trigger": "hcg", "label": "Long agonist protocol, hCG trigger"},
]
N_DOSES, N_REGIMENS = len(DOSES_IU), len(REGIMENS)
N_ARMS = N_DOSES * N_REGIMENS                          # arm = dose_index * N_REGIMENS + regimen_index

CONTEXT = ["age", "amh_ng_ml", "afc", "fsh_iu_l", "bmi", "pcos", "first_cycle", "previous_oocytes", "previous_ohss"]
CONTEXT_RANGES = {"age": (18, 46), "amh_ng_ml": (0.01, 25), "afc": (0, 60), "fsh_iu_l": (0.5, 40), "bmi": (15, 50),
                  "pcos": (0, 1), "first_cycle": (0, 1), "previous_oocytes": (0, 60), "previous_ohss": (0, 1)}


def arm_index(dose_index, regimen_index):
    return np.asarray(dose_index) * N_REGIMENS + np.asarray(regimen_index)


def split_arm(arm):
    arm = np.asarray(arm)
    return arm // N_REGIMENS, arm % N_REGIMENS


def describe_arm(arm: int) -> dict:
    dose_index, regimen_index = int(arm) // N_REGIMENS, int(arm) % N_REGIMENS
    regimen = REGIMENS[regimen_index]
    return {"arm": int(arm), "dose_iu": DOSES_IU[dose_index], "protocol": regimen["protocol"], "trigger": regimen["trigger"],
            "regimen": regimen["key"], "label": f"{DOSES_IU[dose_index]} IU, {regimen['label']}"}
