"""Stimulation cycle simulator with a known dose response model.

Recommending a dose is a counterfactual question: what would have happened to
this patient on a different dose? Real records only ever show one answer per
cycle. A simulator with a documented pharmacodynamic model makes the question
answerable, so every policy in this project can be scored against the true
expected outcome as well as by off policy estimators.

Each patient has two latent traits that no model ever sees:

* ``capacity``: the number of follicles that can be recruited, driven by AFC.
* ``ed50``: the daily dose producing half of that capacity, which rises with
  age and BMI and falls with polycystic ovaries.

Oocyte yield follows a sigmoid Emax curve in dose. OHSS risk rises steeply
with yield and is cut sharply by a GnRH agonist trigger, which is only
available in an antagonist protocol. Historical decisions come from a noisy
clinician heuristic whose choice probabilities are recorded, as a real
logging policy would be for off policy evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stimsafe.data.schema import DOSES_IU, N_ARMS, N_REGIMENS, REGIMENS, arm_index

DOSE = np.array(DOSES_IU, dtype=float)
MATURITY_RATE = 0.78


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_patients(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observable patient context and the latent pharmacodynamic parameters."""
    age = np.clip(rng.normal(34.5, 4.5, n), 21, 45).round(1)
    pcos = (rng.random(n) < 0.14).astype(int)
    log_amh = 1.2 - 0.085 * (age - 30) + 0.95 * pcos + rng.normal(0, 0.6, n)
    amh = np.clip(np.exp(log_amh), 0.05, 20).round(2)
    afc = rng.poisson(np.exp(2.25 + 0.5 * (log_amh - 1) - 0.02 * (age - 30))).clip(1, 50)
    fsh = np.clip(np.exp(1.85 + 0.012 * (age - 30) - 0.22 * (log_amh - 1) + rng.normal(0, 0.25, n)), 1.5, 30).round(1)
    bmi = np.clip(21 + rng.gamma(3.0, 1.6, n) + 2.5 * pcos, 16.5, 45).round(1)

    capacity = 2.2 * afc ** 0.9 * np.exp(rng.normal(0, 0.25, n))
    ed50 = 110 * np.exp(0.03 * (age - 32) + 0.025 * (bmi - 24) - 0.25 * pcos + rng.normal(0, 0.2, n))

    first_cycle = (rng.random(n) < 0.6).astype(int)
    prior_dose = rng.choice(DOSE, n, p=[0.15, 0.4, 0.3, 0.15])
    prior_mean = capacity * prior_dose ** 2 / (ed50 ** 2 + prior_dose ** 2)
    previous_oocytes = np.where(first_cycle == 1, np.nan, rng.poisson(prior_mean))
    previous_ohss = np.where(first_cycle == 1, 0, rng.random(n) < _sigmoid(-6.0 + 0.19 * np.minimum(prior_mean, 35) + 0.5 * pcos)).astype(int)

    context = pd.DataFrame({"age": age, "amh_ng_ml": amh, "afc": afc.astype(float), "fsh_iu_l": fsh, "bmi": bmi, "pcos": pcos,
                            "first_cycle": first_cycle, "previous_oocytes": previous_oocytes, "previous_ohss": previous_ohss})
    latent = pd.DataFrame({"capacity": capacity, "ed50": ed50})
    return context, latent


def expected_oocytes(latent: pd.DataFrame, dose_index, regimen_index) -> np.ndarray:
    dose = DOSE[np.asarray(dose_index)]
    mean = latent["capacity"].to_numpy() * dose ** 2 / (latent["ed50"].to_numpy() ** 2 + dose ** 2)
    return mean * np.where(np.asarray(regimen_index) == 2, 1.06, 1.0)        # long agonist yields slightly more


def ohss_probability(context: pd.DataFrame, oocytes, regimen_index) -> np.ndarray:
    regimen_index = np.asarray(regimen_index)
    logit = (-6.3 + 0.19 * np.minimum(oocytes, 35) + 0.5 * context["pcos"].to_numpy() - 0.04 * (context["age"].to_numpy() - 32)
             - 0.05 * (context["bmi"].to_numpy() - 24) + 0.6 * context["previous_ohss"].to_numpy()
             - 2.2 * (regimen_index == 1) + 0.45 * (regimen_index == 2))
    return _sigmoid(logit)


def utility(mature, ohss, oocytes, dose_index, reward_cfg) -> np.ndarray:
    """Clinical utility of a cycle: reward a good mature oocyte yield, penalise OHSS, cancellation and drug use."""
    mature = np.asarray(mature, dtype=float)
    yield_utility = np.where(mature <= 15, np.minimum(mature / 10.0, 1.0), np.maximum(1.0 - 0.04 * (mature - 15), 0.6))
    return (yield_utility - reward_cfg["ohss_penalty"] * np.asarray(ohss) - reward_cfg["cancellation_penalty"] * (np.asarray(oocytes) < 3)
            - reward_cfg["dose_cost_per_iu"] * DOSE[np.asarray(dose_index)])


def draw_outcomes(context, latent, dose_index, regimen_index, rng, reward_cfg) -> dict:
    """Sample one realised cycle for each patient under the given actions."""
    mean = expected_oocytes(latent, dose_index, regimen_index)
    shape = 8.0
    oocytes = rng.poisson(rng.gamma(shape, mean / shape))                  # negative binomial
    mature = rng.binomial(oocytes, MATURITY_RATE)
    ohss = (rng.random(len(mean)) < ohss_probability(context, oocytes, regimen_index)).astype(int)
    return {"oocytes": oocytes, "mature_oocytes": mature, "ohss": ohss,
            "reward": utility(mature, ohss, oocytes, dose_index, reward_cfg)}


def true_arm_values(context, latent, reward_cfg, draws: int = 24, seed: int = 0) -> dict[str, np.ndarray]:
    """Monte Carlo ground truth for every patient and arm: expected reward, OHSS risk and response rates."""
    rng = np.random.default_rng(seed)
    n = len(context)
    out = {k: np.zeros((n, N_ARMS)) for k in ["reward", "ohss", "poor_response", "optimal_response", "oocytes"]}
    for arm in range(N_ARMS):
        dose_index, regimen_index = np.full(n, arm // N_REGIMENS), np.full(n, arm % N_REGIMENS)
        for _ in range(draws):
            sample = draw_outcomes(context, latent, dose_index, regimen_index, rng, reward_cfg)
            out["reward"][:, arm] += sample["reward"]
            out["ohss"][:, arm] += sample["ohss"]
            out["poor_response"][:, arm] += sample["oocytes"] < 4
            out["optimal_response"][:, arm] += (sample["oocytes"] >= 8) & (sample["oocytes"] <= 18)
            out["oocytes"][:, arm] += sample["oocytes"]
    return {k: v / draws for k, v in out.items()}


def clinician_policy(context: pd.DataFrame) -> np.ndarray:
    """Choice probabilities of the historical prescribing heuristic over all arms."""
    age, amh, pcos = context["age"].to_numpy(), context["amh_ng_ml"].to_numpy(), context["pcos"].to_numpy()
    preferred = np.full(len(context), 1)
    preferred = np.where((amh > 4) | (pcos == 1), 0, preferred)
    preferred = np.where((amh < 2) | (age >= 37), 2, preferred)
    preferred = np.where((amh < 1) | (age >= 40), 3, preferred)
    dose_logits = -1.3 * np.abs(np.arange(len(DOSES_IU))[None, :] - preferred[:, None])
    dose_p = np.exp(dose_logits) / np.exp(dose_logits).sum(axis=1, keepdims=True)
    high = (amh > 3.5) | (pcos == 1)
    antagonist = np.where(high, 0.9, 0.72)
    agonist_trigger = np.where(high, 0.6, 0.08)
    regimen_p = np.column_stack([antagonist * (1 - agonist_trigger), antagonist * agonist_trigger, 1 - antagonist])
    return (dose_p[:, :, None] * regimen_p[:, None, :]).reshape(len(context), N_ARMS)


def simulate_log(n_cycles: int, reward_cfg: dict, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the historical log and, separately, the latent parameters used for ground truth scoring."""
    rng = np.random.default_rng(seed)
    context, latent = sample_patients(n_cycles, rng)
    probabilities = clinician_policy(context)
    arm = (rng.random(n_cycles)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1).clip(0, N_ARMS - 1)
    dose_index, regimen_index = arm // N_REGIMENS, arm % N_REGIMENS
    outcomes = draw_outcomes(context, latent, dose_index, regimen_index, rng, reward_cfg)
    log = context.copy()
    log.insert(0, "cycle_id", np.arange(1, n_cycles + 1))
    log["dose_iu"] = DOSE[dose_index].astype(int)
    log["protocol"] = [REGIMENS[r]["protocol"] for r in regimen_index]
    log["trigger"] = [REGIMENS[r]["trigger"] for r in regimen_index]
    log["arm"] = arm_index(dose_index, regimen_index)
    log["propensity"] = probabilities[np.arange(n_cycles), arm].round(5)
    for key, values in outcomes.items():
        log[key] = values
    latent.insert(0, "cycle_id", log["cycle_id"])
    return log, latent
