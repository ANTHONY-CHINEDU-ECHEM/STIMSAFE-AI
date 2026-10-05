import numpy as np
import pandas as pd

from stimsafe.data.schema import CONTEXT, DOSES_IU, N_ARMS, N_REGIMENS, arm_index, describe_arm, split_arm
from stimsafe.data.simulate import clinician_policy, expected_oocytes, ohss_probability, sample_patients, simulate_log, utility
from stimsafe.policy.safety import SAFE_REGIMEN, allowed_arms, high_risk_mask, risk_reasons


def test_log_is_deterministic_and_propensities_are_valid(small_log, cfg_dict):
    log, _ = small_log
    again, _ = simulate_log(5000, cfg_dict["reward"], seed=3)
    pd.testing.assert_frame_equal(log, again)
    assert log["propensity"].between(0, 1).all() and (log["propensity"] > 0).all()
    assert np.allclose(clinician_policy(log[CONTEXT]).sum(axis=1), 1.0)


def test_arm_encoding_round_trips():
    for arm in range(N_ARMS):
        dose, regimen = split_arm(arm)
        assert arm_index(dose, regimen) == arm
    assert describe_arm(4)["dose_iu"] == DOSES_IU[1] and describe_arm(4)["trigger"] == "gnrh_agonist"


def test_yield_rises_with_dose_and_agonist_trigger_cuts_ohss():
    context, latent = sample_patients(2000, np.random.default_rng(0))
    low = expected_oocytes(latent, np.zeros(2000, int), np.zeros(2000, int))
    high = expected_oocytes(latent, np.full(2000, 3), np.zeros(2000, int))
    assert (high > low).all()
    oocytes = np.full(2000, 22)
    hcg = ohss_probability(context, oocytes, np.zeros(2000, int))
    agonist = ohss_probability(context, oocytes, np.ones(2000, int))
    assert (agonist < hcg / 3).all()


def test_utility_rewards_good_yield_and_penalises_ohss(cfg_dict):
    reward = cfg_dict["reward"]
    good = utility(np.array([12]), np.array([0]), np.array([15]), np.array([1]), reward)[0]
    poor = utility(np.array([1]), np.array([0]), np.array([2]), np.array([1]), reward)[0]
    ohss = utility(np.array([12]), np.array([1]), np.array([15]), np.array([1]), reward)[0]
    assert good > poor and good - ohss == reward["ohss_penalty"]


def test_safety_filter_restricts_flagged_patients(cfg_dict):
    safety = cfg_dict["safety"]
    patients = pd.DataFrame([
        {"age": 29, "amh_ng_ml": 6.0, "afc": 28, "fsh_iu_l": 5, "bmi": 21, "pcos": 1, "first_cycle": 1, "previous_oocytes": np.nan, "previous_ohss": 0},
        {"age": 39, "amh_ng_ml": 0.9, "afc": 6, "fsh_iu_l": 10, "bmi": 27, "pcos": 0, "first_cycle": 1, "previous_oocytes": np.nan, "previous_ohss": 0},
    ])
    assert high_risk_mask(patients, safety).tolist() == [True, False]
    allowed = allowed_arms(patients, safety)
    flagged_arms = np.where(allowed[0])[0]
    assert all(a % N_REGIMENS == SAFE_REGIMEN and DOSES_IU[a // N_REGIMENS] <= safety["max_dose_when_flagged_iu"] for a in flagged_arms)
    assert allowed[1].all()
    assert len(risk_reasons(patients.iloc[0].to_dict(), safety)) >= 3 and risk_reasons(patients.iloc[1].to_dict(), safety) == []


def test_predicted_risk_ceiling_never_leaves_a_patient_without_options(cfg_dict):
    patients = pd.DataFrame([{"age": 33, "amh_ng_ml": 2.0, "afc": 10, "fsh_iu_l": 7, "bmi": 24, "pcos": 0, "first_cycle": 1,
                              "previous_oocytes": np.nan, "previous_ohss": 0}])
    risk = np.full((1, N_ARMS), 0.5)
    risk[0, 7] = 0.2
    allowed = allowed_arms(patients, cfg_dict["safety"], risk)
    assert allowed.sum() == 1 and allowed[0, 7]
