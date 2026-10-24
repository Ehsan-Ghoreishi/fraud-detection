"""Command-line entry point: ``fraud-detection <command>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fraud_detection import config

log = logging.getLogger(__name__)


def _generate(args) -> None:
    from fraud_detection.data import generate
    from fraud_detection.simulation import SimulationConfig

    generate(args.data, SimulationConfig(
        n_customers=args.customers, days=args.days,
        fraud_episode_rate=args.fraud_episode_rate), seed=args.seed)


def _eda(args) -> None:
    from fraud_detection import eda

    eda.main(args.data, args.reports)


def _train(args) -> None:
    from fraud_detection.model import train_and_evaluate
    from fraud_detection.scoring import save_model

    out = train_and_evaluate(data_path=args.data, reports_dir=args.reports,
                             chargeback_fee=args.chargeback_fee,
                             cost_fp=args.cost_fp, cv_folds=args.cv_folds,
                             test_start=args.test_start,
                             label_delay_days=args.label_delay_days,
                             threshold_folds=args.threshold_folds,
                             models=args.models)
    path = save_model(out["model"], out["threshold"],
                      metadata={"best_model": out["summary"]["best_model"],
                                "chargeback_fee": args.chargeback_fee,
                                "cost_false_positive": args.cost_fp},
                      path=args.model, reference=out["reference"])
    log.info("Model saved to %s", path)
    if args.plots:
        from fraud_detection.plots import make_all

        make_all(out, args.reports)
        log.info("Figures written to %s", args.reports)


def _score(args) -> None:
    from fraud_detection.data import load_transactions
    from fraud_detection.scoring import load_model, score

    scored = score(load_transactions(args.input), load_model(args.model))
    scored.to_csv(args.output, index=False)
    log.info("Scored %s transactions, %s flagged for review -> %s",
             f"{len(scored):,}", f"{scored.flag.sum():,}", args.output)


def _ieee_download(args) -> None:
    from fraud_detection.ieee.data import download

    for path in download(args.src):
        log.info("Ready: %s", path)


def _ieee_train(args) -> None:
    from fraud_detection.ieee.pipeline import train_and_evaluate_ieee
    from fraud_detection.plots import make_ieee

    out = train_and_evaluate_ieee(
        src=args.src, reports_dir=args.reports,
        chargeback_fee=args.chargeback_fee, cost_fp=args.cost_fp,
        cv_folds=args.cv_folds, test_start=args.test_start,
        label_delay_days=args.label_delay_days, models=args.models)
    make_ieee(out, args.reports)
    log.info("IEEE-CIS reports written to %s", args.reports)


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

    p = sub.add_parser("generate", help="simulate a transaction log")
    data_arg(p)
    p.add_argument("--customers", type=int, default=2_500)
    p.add_argument("--days", type=int, default=181)
    p.add_argument("--fraud-episode-rate", type=float, default=0.03,
                   help="fraud episodes per customer per month")
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
    p.add_argument("--chargeback-fee", type=float,
                   default=config.CHARGEBACK_FEE,
                   help="€ added to the amount of every missed fraud "
                        "(default: %(default)s)")
    p.add_argument("--cost-fp", type=float, default=config.COST_FALSE_POSITIVE,
                   help="cost of a false alarm in € (default: %(default)s)")
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    p.add_argument("--test-start", default=config.TEST_START,
                   help="first day of the out-of-time test period")
    p.add_argument("--label-delay-days", type=float,
                   default=config.LABEL_DELAY_DAYS,
                   help="days before --test-start whose labels are not yet "
                        "known (default: %(default)s)")
    p.add_argument("--threshold-folds", type=int,
                   default=config.THRESHOLD_FOLDS,
                   help="tune the threshold on this many most recent CV folds")
    p.add_argument("--models", nargs="+", metavar="NAME",
                   help="restrict candidates, e.g. gradient_boosting")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="skip writing report figures")
    p.set_defaults(func=_train)

    from fraud_detection.ieee.data import IEEE_DIR
    from fraud_detection.ieee.pipeline import IEEE_REPORTS, IEEE_TEST_START

    p = sub.add_parser("ieee-download",
                       help="download the real IEEE-CIS data (Kaggle API)")
    p.add_argument("--src", type=Path, default=IEEE_DIR)
    p.set_defaults(func=_ieee_download)

    p = sub.add_parser("ieee-train",
                       help="run the evaluation protocol on IEEE-CIS")
    p.add_argument("--src", type=Path, default=IEEE_DIR)
    p.add_argument("--reports", type=Path, default=IEEE_REPORTS)
    p.add_argument("--chargeback-fee", type=float, default=config.CHARGEBACK_FEE)
    p.add_argument("--cost-fp", type=float, default=config.COST_FALSE_POSITIVE)
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    p.add_argument("--test-start", default=IEEE_TEST_START)
    p.add_argument("--label-delay-days", type=float,
                   default=config.LABEL_DELAY_DAYS)
    p.add_argument("--models", nargs="+", metavar="NAME")
    p.set_defaults(func=_ieee_train)

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
