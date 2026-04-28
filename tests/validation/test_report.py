from pathlib import Path

from validation.report import results_to_df, summarize, write_report
from validation.runner import SeedResult


def _fake_results() -> list[SeedResult]:
    return [
        SeedResult("pbc", 0, 0.84, 0.85, -0.01),
        SeedResult("pbc", 1, 0.85, 0.84, 0.01),
        SeedResult("pbc", 2, 0.83, 0.86, -0.03),
        SeedResult("hd", 0, 0.65, 0.70, -0.05),
        SeedResult("hd", 1, 0.66, 0.71, -0.05),
    ]


def test_summarize_shape_and_columns():
    df = summarize(results_to_df(_fake_results()))
    assert list(df.columns) == [
        "dataset",
        "n_seeds",
        "median_c_crforest",
        "median_c_rfsrc",
        "median_delta_c",
        "iqr_delta_c",
        "max_abs_delta_c",
        "pass",
    ]
    assert set(df["dataset"].tolist()) == {"pbc", "hd"}
    pbc_row = df[df["dataset"] == "pbc"].iloc[0]
    assert pbc_row["n_seeds"] == 3
    assert pbc_row["median_delta_c"] == -0.01
    assert bool(pbc_row["pass"]) is False  # |-0.01| not < 0.01
    hd_row = df[df["dataset"] == "hd"].iloc[0]
    assert bool(hd_row["pass"]) is False


def test_summarize_pass_threshold():
    results = [
        SeedResult("t", 0, 0.8, 0.805, -0.005),
        SeedResult("t", 1, 0.81, 0.811, -0.001),
        SeedResult("t", 2, 0.82, 0.822, -0.002),
    ]
    df = summarize(results_to_df(results))
    assert bool(df.iloc[0]["pass"]) is True


def test_write_report_produces_markdown(tmp_path: Path):
    df = summarize(results_to_df(_fake_results()))
    out = tmp_path / "report.md"
    write_report(df, out, run_date="2026-04-17", commit="abc123", n_seeds=3)
    content = out.read_text()
    assert "# crforest vs randomForestSRC" in content
    assert "| Dataset |" in content
    assert "| pbc |" in content or "pbc " in content
    assert "| hd |" in content or "hd " in content
    assert "2026-04-17" in content
    assert "abc123" in content
