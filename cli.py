"""Command line entry point: ``python -m stimsafe.cli <command>``."""
from __future__ import annotations

import argparse
import json
import logging

from stimsafe.config import load_config, resolve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stimsafe", description="StimSafe AI pipeline")
    parser.add_argument("command", choices=["simulate", "train", "report", "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)

    if args.command in {"simulate", "all"}:
        from stimsafe.data.simulate import simulate_log

        log, latent = simulate_log(cfg.data.n_cycles, dict(cfg.reward), cfg.seed)
        path = resolve(cfg.data.raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(path, index=False)
        latent.to_csv(resolve(cfg.data.oracle_path), index=False)
        logging.info("Wrote %d cycles | OHSS %.2f%% | mean oocytes %.1f", len(log), 100 * log["ohss"].mean(), log["oocytes"].mean())
    if args.command in {"train", "all"}:
        from stimsafe.evaluation.pipeline import run_pipeline

        report = run_pipeline(cfg)
        print(json.dumps({name: round(row["true_value"], 4) for name, row in report["policies"].items()}, indent=2))
    if args.command in {"report", "all"}:
        from stimsafe.evaluation.report import generate_figures

        print("Figures:", ", ".join(generate_figures(cfg)))


if __name__ == "__main__":
    main()
