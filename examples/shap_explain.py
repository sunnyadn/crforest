# /// script
# requires-python = ">=3.11"
# dependencies = ["comprisk", "marimo>=0.23", "matplotlib>=3.8", "numpy"]
# ///
"""TreeSHAP for comprisk competing-risk forests — interactive marimo notebook.

This is a marimo notebook (a plain, git-friendly ``.py`` file). Run it with::

    # interactive editor
    uv run --with marimo --with matplotlib marimo edit examples/shap_explain.py

    # read-only web app
    uv run --with marimo --with matplotlib marimo run examples/shap_explain.py

    # static HTML report
    uv run --with marimo --with matplotlib marimo export html examples/shap_explain.py -o shap_explain.html

Or, with uv's PEP 723 sandbox (deps from the header above)::

    uvx marimo edit --sandbox examples/shap_explain.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # TreeSHAP for competing-risk forests

    `comprisk` computes **exact polynomial-time TreeSHAP** (Lundberg et al. 2018),
    adapted so that each leaf value is a `(n_causes, n_times)` cause-specific CIF
    tensor. `forest.shap_values(X)` returns:

    - **`shap_values`** — shape `(n_samples, n_features, n_times, n_causes)`
    - **`base_value`** — shape `(n_times, n_causes)`, the training-distribution baseline CIF

    with **point-wise additivity**:

    $$\sum_d \mathrm{shap}[s, d, t, c] + \mathrm{base}[t, c] \;\approx\; \mathrm{predict\_cif}(X_s)[c, t].$$

    For a fixed `(time, cause)` slice the array is also drop-in compatible with
    `shap.summary_plot` if you want the upstream beeswarm/waterfall plots.
    """)
    return


@app.cell
def _(mo):
    n_estimators = mo.ui.slider(50, 300, value=100, step=50, label="n_estimators")
    n_explain = mo.ui.slider(5, 50, value=20, step=5, label="subjects to explain")
    mo.hstack([n_estimators, n_explain], justify="start", gap=2)
    return n_estimators, n_explain


@app.cell
def _():
    import numpy as np

    from comprisk import CompetingRiskForest

    return CompetingRiskForest, np


@app.cell
def _(np):
    # 3-state synthetic competing-risks data: feature_0 drives time-to-event,
    # feature_1 modulates the cause distribution; feature_2..9 are noise.
    rng = np.random.default_rng(42)
    n, p = 500, 10
    X = rng.normal(size=(n, p))
    time = rng.exponential(2.0, size=n) + 0.5 * np.abs(X[:, 0]) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    feature_names = [f"feature_{i}" for i in range(p)]
    return X, event, feature_names, time


@app.cell
def _(CompetingRiskForest, X, event, n_estimators, time):
    forest = CompetingRiskForest(
        n_estimators=n_estimators.value, random_state=42, max_depth=8, n_jobs=-1
    ).fit(X, time, event)
    return (forest,)


@app.cell
def _(forest, mo):
    mo.md(
        f"Fitted **{forest.n_estimators} trees**, **{forest.n_causes_} causes**, "
        f"**{len(forest.unique_times_)}** time-grid points."
    )
    return


@app.cell
def _(X, forest, n_explain):
    X_explain = X[: n_explain.value]
    shap_vals, base = forest.shap_values(X_explain)
    return X_explain, base, shap_vals


@app.cell
def _(X_explain, base, forest, np, shap_vals):
    cif_pred = forest.predict_cif(X_explain)
    add_err = float(np.max(np.abs((shap_vals.sum(axis=1) + base).transpose(0, 2, 1) - cif_pred)))
    return add_err, cif_pred


@app.cell
def _(add_err, base, mo, shap_vals):
    mo.md(
        f"`shap_vals.shape = {shap_vals.shape}`  ·  `base.shape = {base.shape}`  \n"
        f"**Additivity max |error| = {add_err:.2e}**  — exact TreeSHAP, so this should be ≈ machine epsilon."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Global feature importance — mean $|\mathrm{SHAP}|$ over subjects × time, per cause
    """)
    return


@app.cell
def _(feature_names, np, shap_vals):
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_vals).mean(axis=(0, 2))  # (n_features, n_causes)
    n_causes = mean_abs.shape[1]
    order = np.argsort(mean_abs.sum(axis=1))[::-1][:10][::-1]
    yy = np.arange(len(order))
    bar_h = 0.8 / n_causes
    fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
    for j in range(n_causes):
        ax_bar.barh(
            yy + (j - (n_causes - 1) / 2) * bar_h,
            mean_abs[order, j],
            height=bar_h,
            label=f"cause {j + 1}",
        )
    ax_bar.set_yticks(yy, [feature_names[i] for i in order])
    ax_bar.set_xlabel("mean(|SHAP value|)")
    ax_bar.legend()
    fig_bar.tight_layout()
    fig_bar
    return (plt,)


@app.cell
def _(mo, shap_vals):
    subject = mo.ui.number(0, shap_vals.shape[0] - 1, value=0, label="subject index")
    subject
    return (subject,)


@app.cell
def _(base, cif_pred, feature_names, forest, mo, np, shap_vals, subject):
    s = min(int(subject.value), shap_vals.shape[0] - 1)
    cause, ti = 0, -1  # cause 1, last time point
    sl = shap_vals[s, :, ti, cause]
    breakdown = "\n".join(
        f"| `{feature_names[i]}` | {sl[i]:+.4f} |" for i in np.argsort(np.abs(sl))[::-1][:6]
    )
    mo.md(
        f"### Subject {s} — cause 1 CIF at t = {forest.unique_times_[ti]:.2f}\n\n"
        f"baseline $\\mathbb{{E}}[\\mathrm{{CIF}}]$ = **{base[ti, cause]:.4f}** &nbsp;·&nbsp; "
        f"predicted CIF = **{cif_pred[s, cause, ti]:.4f}** &nbsp;·&nbsp; "
        f"$\\sum$ attributions = **{sl.sum():.4f}**\n\n"
        f"| feature | SHAP |\n|---|---|\n{breakdown}"
    )
    return cause, s


@app.cell
def _(cause, feature_names, forest, np, plt, s, shap_vals):
    top = np.argsort(np.abs(shap_vals[s, :, :, cause]).mean(axis=1))[::-1][:5]
    fig_t, ax_t = plt.subplots(figsize=(7.5, 4.5))
    for fi in top:
        ax_t.plot(
            forest.unique_times_, shap_vals[s, fi, :, cause], marker="o", ms=3, label=feature_names[fi]
        )
    ax_t.axhline(0, color="k", lw=0.7)
    ax_t.set_xlabel("time")
    ax_t.set_ylabel("SHAP value for cause 1 CIF")
    ax_t.set_title(f"subject {s}: how feature attributions evolve over the time grid")
    ax_t.legend(fontsize=8)
    fig_t.tight_layout()
    fig_t
    return


if __name__ == "__main__":
    app.run()
