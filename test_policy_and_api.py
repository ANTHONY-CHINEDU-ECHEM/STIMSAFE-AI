import numpy as np
import pytest
from fastapi.testclient import TestClient

from stimsafe.api.main import app, recommender_dependency
from stimsafe.data.schema import N_ARMS
from stimsafe.inference.recommender import InvalidPatientError, ProtocolRecommender, normalise_patient
from stimsafe.policy.bandit import LinUCB, off_policy_estimates

HIGH = {"age": 29, "amh_ng_ml": 6.5, "afc": 30, "fsh_iu_l": 5.0, "bmi": 21.5, "pcos": 1, "first_cycle": 1}
LOW = {"age": 41, "amh_ng_ml": 0.5, "afc": 4, "fsh_iu_l": 12.0, "bmi": 27.0, "pcos": 0, "first_cycle": 0, "previous_oocytes": 3, "previous_ohss": 0}


def test_off_policy_estimators_recover_a_known_value():
    rng = np.random.default_rng(0)
    n = 40_000
    true_means = np.linspace(0.1, 0.9, N_ARMS)
    propensity_matrix = rng.dirichlet(np.ones(N_ARMS) * 3, n)
    logged = (rng.random(n)[:, None] > propensity_matrix.cumsum(axis=1)).sum(axis=1).clip(0, N_ARMS - 1)
    reward = true_means[logged] + rng.normal(0, 0.2, n)
    policy = np.full(n, N_ARMS - 1)
    wrong_model = np.full((n, N_ARMS), 0.5)                       # deliberately uninformative reward model
    estimates = off_policy_estimates(policy, logged, reward, propensity_matrix[np.arange(n), logged], wrong_model, rounds=50)
    assert estimates["direct_method"] == pytest.approx(0.5)
    assert estimates["doubly_robust"] == pytest.approx(0.9, abs=0.03)       # corrected by the propensities
    assert estimates["snips"] == pytest.approx(0.9, abs=0.03)


def test_linucb_learns_the_better_arm():
    rng = np.random.default_rng(1)
    bandit = LinUCB(n_features=2, n_arms=2, alpha=0.5)
    picks = []
    for _ in range(600):
        x = np.array([1.0, rng.normal()])
        arm = int(bandit.scores(x).argmax())
        bandit.update(arm, x, (0.8 if arm == 1 else 0.2) + rng.normal(0, 0.1))
        picks.append(arm)
    assert np.mean(picks[-200:]) > 0.95


def test_pipeline_orders_policies_sensibly(pipeline_run):
    _, report = pipeline_run
    value = {name: row["true_value"] for name, row in report["policies"].items()}
    assert value["Oracle (true best arm)"] >= max(v for k, v in value.items() if not k.startswith("Oracle"))
    assert value["StimSafe (bandit with safety filter)"] > value["Clinician heuristic (logged)"]
    assert value["StimSafe (bandit with safety filter)"] > value["Fixed 150 IU for everyone"]
    safe, free = report["policies"]["StimSafe (bandit with safety filter)"], report["policies"]["Bandit policy (unconstrained)"]
    assert safe["true_ohss_rate_high_risk"] <= free["true_ohss_rate_high_risk"] + 1e-9


@pytest.fixture(scope="module")
def recommender(pipeline_run):
    return ProtocolRecommender.load(pipeline_run[0].artifacts.bundle)


def test_high_responder_gets_the_protective_protocol(recommender):
    result = recommender.recommend(HIGH)
    rec = result["recommendation"]
    assert result["safety"]["high_risk"] and len(result["safety"]["reasons"]) >= 3
    assert rec["dose_iu"] <= 150 and rec["protocol"] == "antagonist" and rec["trigger"] == "gnrh_agonist"
    assert sum(o["allowed"] for o in result["options"]) <= 2 and len(result["options"]) == N_ARMS


def test_low_reserve_patient_gets_a_higher_dose_than_high_responder(recommender):
    low, high = recommender.recommend(LOW), recommender.recommend(HIGH)
    assert not low["safety"]["high_risk"]
    assert low["recommendation"]["dose_iu"] > high["recommendation"]["dose_iu"]
    by_dose = sorted((o for o in low["options"] if o["regimen"] == "antagonist_hcg"), key=lambda o: o["dose_iu"])
    assert all(a["expected_oocytes"] <= b["expected_oocytes"] + 1e-6 for a, b in zip(by_dose, by_dose[1:]))     # monotone in dose


def test_similar_patients_summary(recommender):
    summary = recommender.similar_patients(HIGH)
    assert summary["cohort_size"] == 150 and sum(g["cycles"] for g in summary["by_dose"]) == 150


def test_patient_validation():
    assert np.isnan(normalise_patient({**LOW, "first_cycle": 1})["previous_oocytes"])
    with pytest.raises(InvalidPatientError):
        normalise_patient({**HIGH, "age": 70})
    with pytest.raises(InvalidPatientError):
        normalise_patient({k: v for k, v in HIGH.items() if k != "afc"})


def test_api_contract(recommender):
    app.dependency_overrides[recommender_dependency] = lambda: recommender
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    body = client.post("/api/recommend", json=HIGH).json()
    assert body["recommendation"]["allowed"] and body["comparators"]["usual_practice"]["dose_iu"] in {100, 150, 225, 300}
    assert client.post("/api/similar", json=LOW).json()["cohort_size"] == 150
    assert len(client.get("/api/policies").json()["policies"]) == 7
    assert client.post("/api/recommend", json={**HIGH, "age": 12}).status_code == 422
    assert client.post("/api/recommend", json={**HIGH, "extra": 1}).status_code == 422
    app.dependency_overrides.clear()


def test_extreme_responder_is_escalated_for_review(recommender):
    extreme = {"age": 26, "amh_ng_ml": 12.0, "afc": 45, "fsh_iu_l": 4.5, "bmi": 19.0, "pcos": 1, "first_cycle": 0, "previous_oocytes": 35, "previous_ohss": 1}
    result = recommender.recommend(extreme)
    assert result["recommendation"]["dose_iu"] == 100 and result["recommendation"]["trigger"] == "gnrh_agonist"
    assert isinstance(result["safety"]["above_risk_ceiling"], bool)
