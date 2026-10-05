"""Training and evaluation pipeline for every policy."""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from stimsafe import __version__
from stimsafe.config import Config, resolve
from stimsafe.data.schema import CONTEXT, DOSES_IU, N_ARMS, N_REGIMENS, describe_arm
from stimsafe.data.simulate import clinician_policy, draw_outcomes, true_arm_values
from stimsafe.models.outcome import OutcomeModels, ProtocolClassifier, design_matrix
from stimsafe.policy.bandit import bandit_features, off_policy_estimates, run_online_simulation
from stimsafe.policy.safety import allowed_arms, high_risk_mask

logger = logging.getLogger(__name__)


def age_rule_policy(context: pd.DataFrame) -> np.ndarray:
    """A common simple rule: dose by age band, antagonist protocol with hCG trigger for everyone."""
    dose = np.select([context["age"] < 35, context["age"] < 38, context["age"] < 41], [1, 2, 2], default=3)
    return dose * N_REGIMENS


def greedy(values: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    return (values if allowed is None else np.where(allowed, values, -np.inf)).argmax(axis=1)


def similarity_index(log: pd.DataFrame, size: int, seed: int) -> dict:
    """Nearest neighbour index over historical patients, used to show comparable past cycles."""
    sample = log.sample(min(size, len(log)), random_state=seed).reset_index(drop=True)
    X = sample[["age", "amh_ng_ml", "afc", "bmi", "pcos"]].astype(float).copy()
    X["amh_ng_ml"] = np.log(X["amh_ng_ml"])
    mean, std = X.mean().to_numpy(), X.std().to_numpy() + 1e-9
    weights = np.array([1.0, 1.5, 1.5, 0.6, 1.2])
    index = NearestNeighbors().fit((X.to_numpy() - mean) / std * weights)
    return {"index": index, "mean": mean, "std": std, "weights": weights,
            "outcomes": sample[["dose_iu", "protocol", "trigger", "oocytes", "mature_oocytes", "ohss"]]}


def run_pipeline(cfg: Config) -> dict:
    log = pd.read_csv(resolve(cfg.data.raw_path))
    latent = pd.read_csv(resolve(cfg.data.oracle_path))
    rng = np.random.default_rng(cfg.seed)
    is_train = rng.random(len(log)) < cfg.data.split["train"]
    train, test = log[is_train].reset_index(drop=True), log[~is_train].reset_index(drop=True)
    latent_test = latent[~is_train].reset_index(drop=True)
    safety_cfg, reward_cfg = dict(cfg.safety), dict(cfg.reward)

    models = OutcomeModels(cfg.seed).fit(train)
    classifier = ProtocolClassifier(cfg.seed).fit(train)
    logger.info("Outcome models trained on %d cycles; protocol classifier on %d successful cycles", len(train), classifier.n_training_cycles_)

    context = test[CONTEXT]
    predicted = models.predict_all_arms(context)
    truth = true_arm_values(context, latent_test, reward_cfg, seed=cfg.seed)
    allowed_rules = allowed_arms(context, safety_cfg)
    allowed_full = allowed_arms(context, safety_cfg, predicted["ohss"])
    flagged = high_risk_mask(context, safety_cfg)

    policies = {
        "Clinician heuristic (logged)": None,
        "Fixed 150 IU for everyone": np.full(len(test), 1 * N_REGIMENS),
        "Age band rule": age_rule_policy(context),
        "Multi output classifier": classifier.predict_arm(context),
        "Bandit policy (unconstrained)": greedy(predicted["reward"]),
        "StimSafe (bandit with safety filter)": greedy(predicted["reward"], allowed_full),
        "Oracle (true best arm)": greedy(truth["reward"]),
    }
    rows = np.arange(len(test))
    clinician_probs = clinician_policy(context)
    evaluation = {}
    for name, arm in policies.items():
        if arm is None:
            true_stats = {k: float((clinician_probs * truth[k]).sum(axis=1).mean()) for k in truth}
            flagged_ohss = float((clinician_probs * truth["ohss"]).sum(axis=1)[flagged].mean())
            mean_dose = float((clinician_probs.reshape(len(test), len(DOSES_IU), N_REGIMENS).sum(axis=2) * np.array(DOSES_IU)).sum(axis=1).mean())
            ope = {"doubly_robust": float(test["reward"].mean()), "note": "observed mean reward of the log"}
        else:
            true_stats = {k: float(truth[k][rows, arm].mean()) for k in truth}
            flagged_ohss = float(truth["ohss"][rows, arm][flagged].mean())
            mean_dose = float(np.array(DOSES_IU)[arm // N_REGIMENS].mean())
            ope = off_policy_estimates(arm, test["arm"].to_numpy(), test["reward"].to_numpy(), test["propensity"].to_numpy(),
                                       predicted["reward"], seed=cfg.seed)
        evaluation[name] = {
            "true_value": true_stats["reward"], "true_ohss_rate": true_stats["ohss"], "true_ohss_rate_high_risk": flagged_ohss,
            "true_poor_response_rate": true_stats["poor_response"], "true_optimal_response_rate": true_stats["optimal_response"],
            "true_mean_oocytes": true_stats["oocytes"], "mean_dose_iu": mean_dose, "off_policy": ope,
        }
        logger.info("%-38s true value %.4f | OHSS %.2f%% | optimal response %.1f%%", name, true_stats["reward"],
                    100 * true_stats["ohss"], 100 * true_stats["optimal_response"])

    # How good are the outcome models on the logged action?
    X_test = design_matrix(test, test["arm"].to_numpy())
    model_quality = {
        "oocytes_mae": float(mean_absolute_error(test["oocytes"], models.oocytes.predict(X_test))),
        "oocytes_mae_naive": float(mean_absolute_error(test["oocytes"], np.full(len(test), train["oocytes"].mean()))),
        "oocyte_band_coverage": float(((test["oocytes"] >= models.oocytes_low.predict(X_test)) & (test["oocytes"] <= models.oocytes_high.predict(X_test))).mean()),
        "ohss_roc_auc": float(roc_auc_score(test["ohss"], models.ohss.predict_proba(X_test)[:, 1])),
        "counterfactual_oocytes_mae": float(np.abs(predicted["oocytes"] - truth["oocytes"]).mean()),
        "counterfactual_ohss_mae": float(np.abs(predicted["ohss"] - truth["ohss"]).mean()),
    }
    target = ProtocolClassifier.targets(test)
    pred_arm = policies["Multi output classifier"]
    model_quality["classifier_agreement_with_oracle_dose"] = float((pred_arm // N_REGIMENS == policies["Oracle (true best arm)"] // N_REGIMENS).mean())
    model_quality["classifier_agreement_with_logged_dose"] = float((pred_arm // N_REGIMENS == target[:, 0]).mean())

    # Online learning in the simulator.
    n_rounds = min(cfg.bandit.rounds, len(test))
    features, mean, std = bandit_features(context.iloc[:n_rounds])
    sim_rng = np.random.default_rng(cfg.seed + 1)
    context_rounds, latent_rounds = context.iloc[:n_rounds].reset_index(drop=True), latent_test.iloc[:n_rounds].reset_index(drop=True)

    # One realised outcome per patient and arm, drawn up front so the online loop is a table lookup.
    realised = np.column_stack([
        draw_outcomes(context_rounds, latent_rounds, np.full(n_rounds, arm // N_REGIMENS), np.full(n_rounds, arm % N_REGIMENS), sim_rng, reward_cfg)["reward"]
        for arm in range(N_ARMS)])

    def sample_reward(i: int, arm: int) -> float:
        return float(realised[i, arm])

    regret = run_online_simulation(features, sample_reward, truth["reward"][:n_rounds], clinician_probs[:n_rounds],
                                   allowed_rules[:n_rounds], cfg.bandit.linucb_alpha, cfg.bandit.epsilon, cfg.seed)

    stimsafe_arm = policies["StimSafe (bandit with safety filter)"]
    report = {
        "rows": {"train": len(train), "test": len(test)}, "high_risk_share": float(flagged.mean()),
        "logged": {"ohss_rate": float(test["ohss"].mean()), "mean_oocytes": float(test["oocytes"].mean()), "mean_reward": float(test["reward"].mean()),
                   "poor_response_rate": float((test["oocytes"] < 4).mean()), "optimal_response_rate": float(test["oocytes"].between(8, 18).mean())},
        "policies": evaluation, "model_quality": model_quality,
        "dose_shift": pd.crosstab(test["dose_iu"], np.array(DOSES_IU)[stimsafe_arm // N_REGIMENS]).to_dict(),
        "recommended_regimen_share": pd.Series(stimsafe_arm % N_REGIMENS).value_counts(normalize=True).sort_index().to_dict(),
        "safety_filter_changed_share": float((stimsafe_arm != policies["Bandit policy (unconstrained)"]).mean()),
        "online": {name: values[::50].round(3).tolist() for name, values in regret.items()}, "online_rounds": n_rounds,
    }

    report_dir = resolve(cfg.artifacts.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)
    np.savez_compressed(report_dir / "curves.npz", predicted_oocytes=predicted["oocytes"][:400], true_oocytes=truth["oocytes"][:400],
                        predicted_ohss=predicted["ohss"][:400], true_ohss=truth["ohss"][:400],
                        context=context.iloc[:400].to_numpy(dtype=float))
    bundle = {"models": models, "classifier": classifier, "safety": safety_cfg, "version": __version__,
              "similar": similarity_index(train, cfg.similar_patients.index_size, cfg.seed), "neighbours": cfg.similar_patients.neighbours,
              "policy_table": evaluation, "arms": [describe_arm(a) for a in range(N_ARMS)]}
    bundle_path = resolve(cfg.artifacts.bundle)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path, compress=3)
    return report
