from __future__ import annotations

import pytest

from stimsafe.config import Config, load_config
from stimsafe.data.simulate import simulate_log


@pytest.fixture(scope="session")
def cfg_dict():
    return dict(load_config())


@pytest.fixture(scope="session")
def small_log(cfg_dict):
    return simulate_log(5000, cfg_dict["reward"], seed=3)


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, small_log, cfg_dict):
    from stimsafe.evaluation.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("run")
    log, latent = small_log
    log.to_csv(root / "log.csv", index=False)
    latent.to_csv(root / "latent.csv", index=False)
    cfg = {**cfg_dict}
    cfg["data"] = {**cfg["data"], "raw_path": str(root / "log.csv"), "oracle_path": str(root / "latent.csv")}
    cfg["bandit"] = {**cfg["bandit"], "rounds": 600}
    cfg["similar_patients"] = {"index_size": 2000, "neighbours": 150}
    cfg["artifacts"] = {"bundle": str(root / "bundle.joblib"), "report_dir": str(root / "reports"), "figure_dir": str(root / "figures")}
    report = run_pipeline(Config(cfg))
    return Config(cfg), report
