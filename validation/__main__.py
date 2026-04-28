"""CLI for the validation harness.

Subcommands:
  calibrate          — time one seed per dataset and extrapolate
  run --seeds N      — run paired-seed comparison and write results parquet
  report             — re-render the markdown report from a results parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd

from validation.calibrate import calibrate
from validation.config import DATASETS, DEFAULT
from validation.report import results_to_df, summarize, write_report
from validation.runner import SeedResult, run_dataset

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _cmd_calibrate(args: argparse.Namespace) -> None:
    datasets = args.dataset or DATASETS
    timings = calibrate(datasets, config=DEFAULT)
    print(f"{'dataset':<12}{'s/seed':>10}{'x ' + str(args.seeds) + ' seeds':>20}")
    for name, secs in timings.items():
        total = secs * args.seeds
        print(f"{name:<12}{secs:>10.1f}{total / 60:>17.1f} min")
    total_min = sum(timings.values()) * args.seeds / 60
    print(f"{'TOTAL':<12}{'':>10}{total_min:>17.1f} min (serial)")


def _cmd_run(args: argparse.Namespace) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = args.dataset or DATASETS
    seeds = list(range(args.seeds))
    config = replace(
        DEFAULT,
        splitrule=args.splitrule,
        cause=args.cause,
        nsplit=args.nsplit,
        split_ntime=args.split_ntime,
    )
    all_results: list[SeedResult] = []
    for name in datasets:
        print(f"[{name}] running {len(seeds)} seeds...")
        results = run_dataset(
            name, seeds=seeds, config=config, n_jobs=args.n_jobs, compare=args.compare
        )
        all_results.extend(results)
    df = results_to_df(all_results)
    run_date = dt.date.today().isoformat()
    parquet_path = REPORTS_DIR / f"{run_date}-results.parquet"
    df.to_parquet(parquet_path, index=False)
    summary = summarize(df)
    md_path = REPORTS_DIR / f"{run_date}-report.md"
    write_report(summary, md_path, run_date=run_date, commit=_git_sha(), n_seeds=args.seeds)
    print(f"wrote {parquet_path}")
    print(f"wrote {md_path}")


def _cmd_report(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.results)
    summary = summarize(df)
    md_path = Path(args.out) if args.out else args.results.with_suffix(".md")
    n_seeds = int(df.groupby("dataset").size().max())
    write_report(
        summary,
        md_path,
        run_date=dt.date.today().isoformat(),
        commit=_git_sha(),
        n_seeds=n_seeds,
    )
    print(f"wrote {md_path}")


def _cmd_bench_vimp(args: argparse.Namespace) -> None:
    from validation.bench_vimp import run

    result = run(
        dataset=args.dataset,
        n=args.n,
        n_repeats=args.n_repeats,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    for k, v in result.items():
        print(f"{k}: {v}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="validation", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Time 1 seed per dataset")
    p_cal.add_argument("--dataset", action="append", help="restrict to named dataset")
    p_cal.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT.n_seeds,
        help="seeds to extrapolate to (default: %(default)s)",
    )
    p_cal.set_defaults(func=_cmd_calibrate)

    p_run = sub.add_parser("run", help="Run paired-seed validation")
    p_run.add_argument("--dataset", action="append", help="restrict to named dataset")
    p_run.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT.n_seeds,
        help="number of seeds (default: %(default)s)",
    )
    p_run.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (default: -1)")
    p_run.add_argument(
        "--compare",
        choices=["rfsrc", "reference"],
        default="rfsrc",
        help="baseline to compare crforest against (default: rfsrc)",
    )
    p_run.add_argument(
        "--splitrule",
        choices=["logrankCR", "logrank"],
        default="logrankCR",
        help="split rule for crforest and rfsrc baseline selection (default: logrankCR)",
    )
    p_run.add_argument(
        "--cause",
        type=int,
        default=DEFAULT.cause,
        help="cause index for C-index scoring and logrank baseline (default: %(default)s)",
    )
    p_run.add_argument(
        "--nsplit",
        type=int,
        default=DEFAULT.nsplit,
        help="random split-point draws per feature per node (default: %(default)s)",
    )
    p_run.add_argument(
        "--split-ntime",
        type=lambda s: None if s.lower() == "none" else int(s),
        default=DEFAULT.split_ntime,
        help="Split-time coarse bins for mode='default'. 'None' (default) = full grid, no coarsening. Pass e.g. 30 to activate.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="Re-render markdown from results parquet")
    p_report.add_argument("results", type=Path, help="results parquet path")
    p_report.add_argument("--out", type=Path, help="markdown output path")
    p_report.set_defaults(func=_cmd_report)

    p_bench = sub.add_parser("bench-vimp", help="Time VIMP on large synthetic workload")
    p_bench.add_argument("--dataset", default="synthetic")
    p_bench.add_argument("--n", type=int, default=100_000)
    p_bench.add_argument("--n-repeats", type=int, default=5)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (default: -1)")
    p_bench.set_defaults(func=_cmd_bench_vimp)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
