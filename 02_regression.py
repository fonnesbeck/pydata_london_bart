# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pymc>=5.28",
#     "pymc-bart>=0.11",
#     "arviz>=0.23",
#     "numpy>=2",
#     "matplotlib>=3.10",
#     "polars>=1.0",
#     "fastf1>=3.8",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # BART for regression

        We move from the from-scratch sampler in notebook 1 to `pymc-bart`,
        the production library that wraps the same algorithm with PyMC's
        sampler and posterior tooling. Two demos:

        1. **Friedman synthetic data** — known ground truth, lets us check
           that BART recovers $f(x)$ with calibrated uncertainty.
        2. **Formula 1 lap times** — real data, where uncertainty matters
           for risk-aware decisions.

        For each fit we plot the posterior predictive HDI band, compute
        out-of-sample coverage, rank variable importance, and visualise
        partial dependence.
        """
    )
    return


@app.cell
def _():
    # pymc-bart's BART RV creates a multiprocessing.Manager(); force the
    # "fork" start method so this works under both marimo edit and bare
    # `python` script execution (Python 3.14 defaults to forkserver, which
    # re-imports the script and breaks without an `if __name__` guard).
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import pymc_bart as pmb

    rng = np.random.default_rng(20260423)
    return Path, np, os, pl, plt, pm, pmb, rng


@app.cell
def _(np, rng):
    def friedman(X, noise=0.0, rng=None):
        y = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )
        if noise > 0:
            assert rng is not None
            y = y + rng.normal(0, noise, size=y.shape[0])
        return y

    n_train, n_feat = 100, 10
    X_fried = rng.uniform(size=(n_train, n_feat))
    y_fried = friedman(X_fried, noise=1.0, rng=rng)
    X_fried_test = rng.uniform(size=(200, n_feat))
    y_fried_test_true = friedman(X_fried_test, noise=0.0)
    return X_fried, X_fried_test, y_fried, y_fried_test_true


@app.cell
def _(X_fried, pm, pmb, y_fried):
    # The pymc-bart API: declare the BART random variable inside a PyMC
    # model alongside any other priors. Wrap X in pm.Data so we can swap it
    # later for out-of-sample predictions.
    with pm.Model() as model_fried:
        X_data = pm.Data("X_data", X_fried)
        mu_bart = pmb.BART("mu", X=X_data, Y=y_fried, m=50)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", mu=mu_bart, sigma=sigma, observed=y_fried, shape=mu_bart.shape)
        idata_fried = pm.sample(
            draws=300,
            tune=300,
            chains=2,
            cores=1,
            random_seed=20260423,
            progressbar=False,
        )
    return idata_fried, model_fried, mu_bart


@app.cell(hide_code=True)
def _(idata_fried, np, plt, y_fried):
    # In-sample fit + sigma posterior. The HDI band is the 90% credible
    # interval of f(x) at each training point, marginalised over posterior
    # trees. The Friedman truth has sigma = 1; we expect the posterior to
    # cover that.
    _mu = idata_fried.posterior["mu"].stack(sample=("chain", "draw")).values
    _f_mean = _mu.mean(axis=1)
    _f_lo, _f_hi = np.quantile(_mu, [0.05, 0.95], axis=1)
    _sigma = idata_fried.posterior["sigma"].values.ravel()

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.0))
    _order = np.argsort(_f_mean)
    _a.errorbar(
        _f_mean[_order],
        y_fried[_order],
        yerr=[(_f_mean - _f_lo)[_order], (_f_hi - _f_mean)[_order]],
        fmt="o",
        ms=3,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.55,
        elinewidth=0.7,
    )
    _lim = (
        min(_f_mean.min(), y_fried.min()) - 1,
        max(_f_mean.max(), y_fried.max()) + 1,
    )
    _a.plot(_lim, _lim, "--", color="C3", lw=1)
    _a.set_xlim(_lim)
    _a.set_ylim(_lim)
    _a.set_xlabel(r"posterior mean $\hat f(x)$ (90% HDI)")
    _a.set_ylabel("observed $y$")
    _a.set_title("Friedman in-sample fit")

    _b.hist(_sigma, bins=30, color="#4c72b0", alpha=0.8, edgecolor="white")
    _b.axvline(1.0, color="C3", ls="--", lw=1, label=r"true $\sigma = 1$")
    _b.set_xlabel(r"$\sigma$")
    _b.set_ylabel("posterior density")
    _b.set_title(rf"$\sigma$ posterior  (mean={_sigma.mean():.2f})")
    _b.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(X_fried_test, idata_fried, model_fried, pm):
    # Out-of-sample predictions: swap the X data, then sample posterior
    # predictive. pmb.BART evaluates the posterior trees at the new X.
    with model_fried:
        pm.set_data({"X_data": X_fried_test})
        pp_fried = pm.sample_posterior_predictive(
            idata_fried,
            var_names=["mu"],
            predictions=True,
            random_seed=20260423,
            progressbar=False,
        )
    return (pp_fried,)


@app.cell(hide_code=True)
def _(np, plt, pp_fried, y_fried_test_true):
    _mu_pred = pp_fried.predictions["mu"].stack(sample=("chain", "draw")).values
    _f_mean = _mu_pred.mean(axis=1)
    _f_lo, _f_hi = np.quantile(_mu_pred, [0.05, 0.95], axis=1)
    _cov = ((_f_lo <= y_fried_test_true) & (y_fried_test_true <= _f_hi)).mean()

    _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
    _order = np.argsort(y_fried_test_true)
    _ax.errorbar(
        y_fried_test_true[_order],
        _f_mean[_order],
        yerr=[(_f_mean - _f_lo)[_order], (_f_hi - _f_mean)[_order]],
        fmt="o",
        ms=3,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.55,
        elinewidth=0.7,
    )
    _lim = (y_fried_test_true.min() - 1, y_fried_test_true.max() + 1)
    _ax.plot(_lim, _lim, "--", color="C3", lw=1)
    _ax.set_xlim(_lim)
    _ax.set_ylim(_lim)
    _ax.set_xlabel(r"true $f(x)$ on held-out test")
    _ax.set_ylabel(r"posterior mean $\hat f(x)$ with 90% HDI")
    _ax.set_title(f"Friedman out-of-sample coverage: {_cov:.0%}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(X_fried, idata_fried, model_fried, mu_bart, pm, pmb):
    # Variable importance: pymc-bart computes a restricted-model R^2 by
    # progressively adding variables in order of inclusion frequency. The
    # Friedman DGP only depends on X_0..X_4; the curve should plateau at
    # k = 5. We reset X_data first because the OOS-prediction cell may
    # have mutated it.
    with model_fried:
        pm.set_data({"X_data": X_fried})
        _vi = pmb.compute_variable_importance(idata_fried, mu_bart, X_fried)
    pmb.plot_variable_importance(_vi)
    return


@app.cell(hide_code=True)
def _(X_fried, model_fried, mu_bart, pm, pmb, y_fried):
    # Partial dependence: marginal effect of each covariate, holding the
    # others at their data distribution. The Friedman DGP only uses
    # X_0..X_4, so X_5..X_9 should produce near-flat curves.
    with model_fried:
        pm.set_data({"X_data": X_fried})
        pmb.plot_pdp(bartrv=mu_bart, X=X_fried, Y=y_fried)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Real data: Formula 1 lap times

        The Friedman fit is calibration. Now a real-data application:
        predicting lap times at the 2024 British Grand Prix from in-race
        progress (tyre life, lap number, stint), weather (air/track
        temperature, humidity, wind), and tyre compound. Silverstone 2024
        had a wet phase, so the dataset includes INTERMEDIATE tyres
        alongside the dry SOFT/MEDIUM/HARD compounds.
        """
    )
    return


@app.cell
def _(Path, os, pl):
    def load_f1_laps():
        override = os.environ.get("F1_LAPS_CSV")
        candidates = []
        if override:
            candidates.append(Path(override))
        candidates.extend(
            [
                Path.cwd() / "data" / "f1_laps.csv",
                Path.home() / "repos" / "pydata_london_bart" / "data" / "f1_laps.csv",
            ]
        )
        for p in candidates:
            if p.exists():
                return pl.read_csv(p)
        raise FileNotFoundError(
            "f1_laps.csv not found. Set F1_LAPS_CSV env var or place the file "
            "at ./data/f1_laps.csv. Regenerate with scripts/pull_f1_laps.py "
            "(requires fastf1 + an internet connection on first run)."
        )

    f1_df = load_f1_laps()
    f1_df.shape
    return (f1_df,)


@app.cell
def _(f1_df, np):
    _num_cols = [
        "tyre_life",
        "lap_number",
        "stint",
        "air_temp",
        "track_temp",
        "humidity",
        "wind_speed",
    ]
    _compounds_in_data = ["SOFT", "HARD", "INTERMEDIATE"]
    f1_feature_names = list(_num_cols) + [f"compound_{c}" for c in _compounds_in_data]

    _X_num = f1_df.select(_num_cols).to_numpy().astype(float)
    _X_cmp = np.column_stack(
        [(f1_df["compound"].to_numpy() == c).astype(float) for c in _compounds_in_data]
    )
    _X = np.concatenate([_X_num, _X_cmp], axis=1)
    _y = f1_df["lap_time_s"].to_numpy().astype(float)

    _n_train = 500
    _n_test = 200
    _rng_f1 = np.random.default_rng(20260423)
    _perm = _rng_f1.permutation(_X.shape[0])
    X_train = _X[_perm[:_n_train]]
    y_train = _y[_perm[:_n_train]]
    X_test = _X[_perm[_n_train : _n_train + _n_test]]
    y_test = _y[_perm[_n_train : _n_train + _n_test]]
    return X_test, X_train, f1_feature_names, y_test, y_train


@app.cell
def _(X_train, pm, pmb, y_train):
    with pm.Model() as model_f1:
        X_data_f1 = pm.Data("X_data", X_train)
        mu_f1 = pmb.BART("mu", X=X_data_f1, Y=y_train, m=50)
        sigma_f1 = pm.HalfNormal("sigma", float(y_train.std()))
        pm.Normal("y", mu=mu_f1, sigma=sigma_f1, observed=y_train, shape=mu_f1.shape)
        idata_f1 = pm.sample(
            draws=300,
            tune=300,
            chains=2,
            cores=1,
            random_seed=20260423,
            progressbar=False,
        )
    return idata_f1, model_f1, mu_f1


@app.cell(hide_code=True)
def _(idata_f1, np, plt, y_train):
    _mu = idata_f1.posterior["mu"].stack(sample=("chain", "draw")).values
    _f_mean = _mu.mean(axis=1)
    _f_lo, _f_hi = np.quantile(_mu, [0.05, 0.95], axis=1)
    _sigma = idata_f1.posterior["sigma"].values.ravel()

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.0))
    _order = np.argsort(_f_mean)
    _a.errorbar(
        _f_mean[_order],
        y_train[_order],
        yerr=[(_f_mean - _f_lo)[_order], (_f_hi - _f_mean)[_order]],
        fmt="o",
        ms=3,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.5,
        elinewidth=0.6,
    )
    _lim = (
        min(_f_mean.min(), y_train.min()) - 0.5,
        max(_f_mean.max(), y_train.max()) + 0.5,
    )
    _a.plot(_lim, _lim, "--", color="C3", lw=1)
    _a.set_xlim(_lim)
    _a.set_ylim(_lim)
    _a.set_xlabel(r"posterior mean lap time $\hat f(x)$ (90% HDI)")
    _a.set_ylabel("observed lap time (s)")
    _a.set_title("F1 in-sample fit")

    _b.hist(_sigma, bins=30, color="#4c72b0", alpha=0.8, edgecolor="white")
    _b.set_xlabel(r"$\sigma$ (s)")
    _b.set_ylabel("posterior density")
    _b.set_title(rf"residual SD posterior  (mean={_sigma.mean():.2f}s)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(X_test, idata_f1, model_f1, pm):
    with model_f1:
        pm.set_data({"X_data": X_test})
        pp_f1 = pm.sample_posterior_predictive(
            idata_f1,
            var_names=["mu"],
            predictions=True,
            random_seed=20260423,
            progressbar=False,
        )
    return (pp_f1,)


@app.cell(hide_code=True)
def _(np, plt, pp_f1, y_test):
    _mu_pred = pp_f1.predictions["mu"].stack(sample=("chain", "draw")).values
    _f_mean = _mu_pred.mean(axis=1)
    _f_lo, _f_hi = np.quantile(_mu_pred, [0.05, 0.95], axis=1)
    _cov = ((_f_lo <= y_test) & (y_test <= _f_hi)).mean()

    _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
    _order = np.argsort(y_test)
    _ax.errorbar(
        y_test[_order],
        _f_mean[_order],
        yerr=[(_f_mean - _f_lo)[_order], (_f_hi - _f_mean)[_order]],
        fmt="o",
        ms=3,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.55,
        elinewidth=0.7,
    )
    _lim = (y_test.min() - 0.5, y_test.max() + 0.5)
    _ax.plot(_lim, _lim, "--", color="C3", lw=1)
    _ax.set_xlim(_lim)
    _ax.set_ylim(_lim)
    _ax.set_xlabel("observed lap time (s)")
    _ax.set_ylabel(r"posterior mean $\hat f(x)$ with 90% HDI")
    _ax.set_title(f"F1 out-of-sample coverage: {_cov:.0%}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(X_train, f1_feature_names, idata_f1, model_f1, mu_f1, pm, pmb):
    with model_f1:
        pm.set_data({"X_data": X_train})
        _vi_f1 = pmb.compute_variable_importance(idata_f1, mu_f1, X_train)
    pmb.plot_variable_importance(_vi_f1, labels=f1_feature_names)
    return


@app.cell(hide_code=True)
def _(X_train, model_f1, mu_f1, pm, pmb, y_train):
    with model_f1:
        pm.set_data({"X_data": X_train})
        pmb.plot_pdp(bartrv=mu_f1, X=X_train, Y=y_train)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
