"""Figures for the README, rebuilt from saved metrics."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stimsafe.config import Config, resolve
from stimsafe.data.schema import CONTEXT, DOSES_IU, N_REGIMENS

INK, PETROL, OCHRE, BRICK, MOSS, FOG = "#12303B", "#1F6F8B", "#C98A1B", "#B5352A", "#2F7F6D", "#A9B4BA"
SHORT = {"Clinician heuristic (logged)": "Clinician heuristic", "Fixed 150 IU for everyone": "Fixed 150 IU", "Age band rule": "Age band rule",
         "Multi output classifier": "Multi output classifier", "Bandit policy (unconstrained)": "Bandit, unconstrained",
         "StimSafe (bandit with safety filter)": "StimSafe", "Oracle (true best arm)": "Oracle"}


def _style() -> None:
    plt.rcParams.update({"savefig.dpi": 160, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.grid": True, "grid.alpha": 0.25})


def generate_figures(cfg: Config) -> list[str]:
    _style()
    report_dir, figure_dir = resolve(cfg.artifacts.report_dir), resolve(cfg.artifacts.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((report_dir / "metrics.json").read_text())
    curves = np.load(report_dir / "curves.npz")
    policies = report["policies"]
    names = list(policies)
    written = []

    def colour(name: str) -> str:
        return PETROL if name.startswith("StimSafe") else OCHRE if name.startswith("Oracle") else FOG

    # 1. Policy value: ground truth beside the doubly robust estimate.
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    y = np.arange(len(names))
    truth = [policies[n]["true_value"] for n in names]
    ax.barh(y, truth, color=[colour(n) for n in names], height=0.6)
    for yi, name in enumerate(names):
        ope = policies[name]["off_policy"]
        if "doubly_robust_ci" in ope:
            low, high = ope["doubly_robust_ci"]
            ax.errorbar(ope["doubly_robust"], yi, xerr=[[ope["doubly_robust"] - low], [high - ope["doubly_robust"]]], fmt="D", color=INK, ms=5, capsize=3)
        ax.text(truth[yi] + 0.004, yi + 0.33, f"{truth[yi]:.3f}", fontsize=9, va="center")
    ax.set_yticks(y, [SHORT[n] for n in names]); ax.invert_yaxis()
    ax.set(xlabel="Expected clinical utility per cycle", xlim=(min(truth) - 0.08, max(truth) + 0.05),
           title="True policy value (bars) and doubly robust estimate (diamonds)")
    fig.tight_layout(); fig.savefig(figure_dir / "policy_value.png"); plt.close(fig); written.append("policy_value.png")

    # 2. Safety and efficacy trade off.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, key, title in zip(axes, ["true_ohss_rate", "true_optimal_response_rate", "true_poor_response_rate"],
                              ["OHSS rate (%)", "Optimal response, 8 to 18 oocytes (%)", "Poor response, under 4 oocytes (%)"]):
        values = [100 * policies[n][key] for n in names]
        ax.barh(y, values, color=[colour(n) for n in names], height=0.6)
        for yi, v in enumerate(values):
            ax.text(v + max(values) * 0.01, yi, f"{v:.1f}", va="center", fontsize=9)
        ax.set_yticks(y, [SHORT[n] for n in names] if ax is axes[0] else []); ax.invert_yaxis(); ax.set_title(title)
        ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout(); fig.savefig(figure_dir / "safety_efficacy.png"); plt.close(fig); written.append("safety_efficacy.png")

    # 3. Learned dose response against the simulator truth for three archetypes.
    context = pd.DataFrame(curves["context"], columns=CONTEXT)
    picks = {"Low reserve": (context["amh_ng_ml"] < 0.8) & (context["age"] > 38), "Typical": context["amh_ng_ml"].between(2, 3) & (context["pcos"] == 0),
             "High responder": (context["pcos"] == 1) & (context["amh_ng_ml"] > 5)}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex=True)
    regimen = 0
    arms = [d * N_REGIMENS + regimen for d in range(len(DOSES_IU))]
    for col, (label, mask) in enumerate(picks.items()):
        index = np.where(mask.to_numpy())[0][:25]
        for row, (pred, true, unit) in enumerate([("predicted_oocytes", "true_oocytes", "Oocytes retrieved"), ("predicted_ohss", "true_ohss", "OHSS risk (%)")]):
            scale = 100 if row == 1 else 1
            ax = axes[row, col]
            ax.plot(DOSES_IU, scale * curves[true][index][:, arms].mean(axis=0), "o--", color=INK, label="Simulator truth")
            ax.plot(DOSES_IU, scale * curves[pred][index][:, arms].mean(axis=0), "s-", color=PETROL, label="Learned model")
            if row == 0:
                ax.axhspan(8, 18, color=MOSS, alpha=0.1, lw=0)
                ax.set_title(f"{label} patients")
            else:
                ax.axhline(100 * cfg.safety.max_predicted_ohss_risk, color=BRICK, ls=":", lw=1)
                ax.set_xlabel("Starting dose (IU per day)")
            if col == 0:
                ax.set_ylabel(unit)
            ax.set_xticks(DOSES_IU)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figure_dir / "dose_response.png"); plt.close(fig); written.append("dose_response.png")

    # 4. Online learning regret.
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    rounds = np.arange(len(next(iter(report["online"].values())))) * 50
    for (name, values), c in zip(report["online"].items(), [PETROL, OCHRE, MOSS, FOG]):
        ax.plot(rounds, values, color=c, lw=2, label=name)
    ax.set(xlabel="Cycles treated", ylabel="Cumulative regret against the best arm", title="Online contextual bandits in the simulator")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "online_regret.png"); plt.close(fig); written.append("online_regret.png")

    # 5. How StimSafe moves doses relative to what was prescribed.
    shift = pd.DataFrame(report["dose_shift"]).reindex(index=[str(d) for d in DOSES_IU]).fillna(0)
    shift.columns = [int(float(c)) for c in shift.columns]
    shift = shift.reindex(columns=DOSES_IU).fillna(0)
    share = shift.div(shift.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(DOSES_IU)):
        for j in range(len(DOSES_IU)):
            ax.text(j, i, f"{100 * share.iloc[i, j]:.0f}%", ha="center", va="center", color="white" if share.iloc[i, j] > 0.5 else INK, fontsize=10)
    ax.set_xticks(range(4), DOSES_IU); ax.set_yticks(range(4), DOSES_IU); ax.grid(False)
    ax.set(xlabel="StimSafe recommended dose (IU)", ylabel="Dose actually prescribed (IU)", title="Where the recommendation differs")
    fig.tight_layout(); fig.savefig(figure_dir / "dose_shift.png"); plt.close(fig); written.append("dose_shift.png")

    # 6. Off policy estimator accuracy.
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for estimator, marker, c in [("direct_method", "s", FOG), ("snips", "^", OCHRE), ("doubly_robust", "D", PETROL)]:
        xs = [policies[n]["true_value"] for n in names if estimator in policies[n]["off_policy"]]
        ys = [policies[n]["off_policy"][estimator] for n in names if estimator in policies[n]["off_policy"]]
        ax.scatter(xs, ys, marker=marker, color=c, s=55, label=estimator.replace("_", " ").replace("snips", "self normalised IPS").capitalize(), zorder=3)
    lims = [min(truth) - 0.03, max(truth) + 0.03]
    ax.plot(lims, lims, color=INK, lw=0.8)
    ax.set(xlabel="True policy value", ylabel="Estimated from logged data", title="Can logs alone rank the policies?", xlim=lims)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figure_dir / "ope_accuracy.png"); plt.close(fig); written.append("ope_accuracy.png")
    _ = joblib
    return written
