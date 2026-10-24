"""Command-line entry point: ``fraud-detection <command>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fraud_detection import config

log = logging.getLogger(__name__)


def _generate(args) -> None:
    from fraud_detection.data import generate_transactions

    df = generate_transactions(n=args.n, fraud_rate=args.fraud_rate,
                               seed=args.seed)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.data, index=False)
    log.info("Generated %s transactions (%.2f%% fraud) -> %s",
             f"{len(df):,}", df.is_fraud.mean() * 100, args.data)


def _eda(args) -> None:
    from fraud_detection import eda

    eda.main(args.data, args.reports)


def _train(args) -> None:
    from fraud_detection.model import train_and_evaluate
    from fraud_detection.scoring import save_model

    out = train_and_evaluate(data_path=args.data, reports_dir=args.reports,
                             cost_fn=args.cost_fn, cost_fp=args.cost_fp,
                             cv_folds=args.cv_folds)
    path = save_model(out["model"], out["threshold"],
                      metadata={"best_model": out["summary"]["best_model"],
                                "cost_false_negative": args.cost_fn,
                                "cost_false_positive": args.cost_fp},
                      path=args.model)
    log.info("Model saved to %s", path)
    if args.plots:
        from fraud_detection.plots import make_all

        make_all(out, args.reports)
        log.info("Figures written to %s", args.reports)


def _score(args) -> None:
    from fraud_detection.scoring import load_model, score

    scored = score(pd.read_csv(args.input), load_model(args.model))
    scored.to_csv(args.output, index=False)
    log.info("Scored %s transactions, %s flagged for review -> %s",
             f"{len(scored):,}", f"{scored.flag.sum():,}", args.output)


def build_parser() -> argparse.ArgumentParser:
    from fraud_detection.scoring import MODEL_PATH

    parser = argparse.ArgumentParser(
        prog="fraud-detection",
        description="Cost-aware fraud detection pipeline.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="only log warnings and errors")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_arg(p):
        p.add_argument("--data", type=Path, default=config.DATA_PATH,
                       help="transactions CSV (default: %(default)s)")

    p = sub.add_parser("generate", help="create the synthetic dataset")
    data_arg(p)
    p.add_argument("--n", type=int, default=60_000)
    p.add_argument("--fraud-rate", type=float, default=0.017)
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    p.set_defaults(func=_generate)

    p = sub.add_parser("eda", help="exploratory figure and summary")
    data_arg(p)
    p.add_argument("--reports", type=Path, default=config.REPORTS)
    p.set_defaults(func=_eda)

    p = sub.add_parser("train", help="select, tune, evaluate and save a model")
    data_arg(p)
    p.add_argument("--reports", type=Path, default=config.REPORTS)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--cost-fn", type=float, default=config.COST_FALSE_NEGATIVE,
                   help="cost of a missed fraud in € (default: %(default)s)")
    p.add_argument("--cost-fp", type=float, default=config.COST_FALSE_POSITIVE,
                   help="cost of a false alarm in € (default: %(default)s)")
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="skip writing report figures")
    p.set_defaults(func=_train)

    p = sub.add_parser("score", help="score a CSV with a saved model")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.set_defaults(func=_score)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
