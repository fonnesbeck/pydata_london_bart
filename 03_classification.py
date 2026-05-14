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
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # BART for classification

        Binary classification with BART means modelling
        $\Pr(Y = 1 \mid x) = \Phi\bigl(g(x)\bigr)$, where $g$ is a sum of
        trees and $\Phi$ is the standard-normal CDF (the probit link). The
        BART prior shrinks $g$ toward zero, which corresponds to a
        baseline probability of $1/2$ — useful regularisation when the
        signal is weak.

        Two demos:

        1. **Synthetic binary data** with known true probabilities — lets
           us check calibration directly.
        2. **GSS 2022 ordinal outcome** — job satisfaction modelled with
           an ordered probit, demonstrating that BART composes naturally
           with PyMC's other likelihoods.
        """
    )
    return


@app.cell
def _():
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    import os
    from pathlib import Path

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import pymc_bart as pmb

    rng = np.random.default_rng(20260423)
    return Path, az, np, os, pl, plt, pm, pmb, rng


@app.cell
def _(np, rng):
    def simulate_binary(n, p, rng):
        X = rng.uniform(-1, 1, size=(n, p))
        logit = 2.5 * X[:, 0] * X[:, 1] + 3 * (X[:, 2] > 0.2) - 0.5
        prob_true = 1 / (1 + np.exp(-logit))
        y = rng.binomial(1, prob_true)
        return X, y, prob_true

    X_cls, y_cls, prob_true = simulate_binary(500, 20, rng)
    X_cls_test, y_cls_test, prob_test_true = simulate_binary(300, 20, rng)
    f"train active rate: {y_cls.mean():.2%}"
    return X_cls, X_cls_test, prob_test_true, y_cls, y_cls_test


@app.cell
def _(X_cls, pm, pmb, y_cls):
    # Probit BART: BART output enters Bernoulli through invprobit. We wrap
    # X in pm.Data so we can swap it for out-of-sample predictions.
    with pm.Model() as model_cls:
        X_data = pm.Data("X_data", X_cls)
        eta = pmb.BART("eta", X=X_data, Y=y_cls.astype(float), m=50)
        p = pm.Deterministic("p", pm.math.invprobit(eta))
        pm.Bernoulli("y", p=p, observed=y_cls, shape=p.shape)
        idata_cls = pm.sample(
            draws=300,
            tune=300,
            chains=2,
            cores=1,
            random_seed=20260423,
            progressbar=False,
        )
    return eta, idata_cls, model_cls


@app.cell
def _(X_cls_test, idata_cls, model_cls, pm):
    with model_cls:
        pm.set_data({"X_data": X_cls_test})
        pp_cls = pm.sample_posterior_predictive(
            idata_cls,
            var_names=["p"],
            predictions=True,
            random_seed=20260423,
            progressbar=False,
        )
    return (pp_cls,)


@app.cell(hide_code=True)
def _(np, plt, pp_cls, prob_test_true, y_cls_test):
    _p_draws = pp_cls.predictions["p"].stack(sample=("chain", "draw")).values
    _p_mean = _p_draws.mean(axis=1)
    _p_lo, _p_hi = np.quantile(_p_draws, [0.05, 0.95], axis=1)

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.0))
    _order = np.argsort(prob_test_true)
    _a.errorbar(
        prob_test_true[_order],
        _p_mean[_order],
        yerr=[(_p_mean - _p_lo)[_order], (_p_hi - _p_mean)[_order]],
        fmt=".",
        ms=3,
        alpha=0.5,
        elinewidth=0.7,
        color="#4c72b0",
    )
    _a.plot([0, 1], [0, 1], "--", color="C3", lw=1)
    _a.set_xlabel(r"true $\Pr(Y=1 \mid x)$")
    _a.set_ylabel("posterior mean (90% CI)")
    _a.set_title("Calibration on the test set")

    _top = np.argsort(-_p_mean)[:20]
    _pos = np.arange(len(_top))
    _colors = ["C2" if y_cls_test[i] == 1 else "C3" for i in _top]
    _b.errorbar(
        _pos,
        _p_mean[_top],
        yerr=[(_p_mean - _p_lo)[_top], (_p_hi - _p_mean)[_top]],
        fmt="none",
        ecolor="#888",
        elinewidth=0.8,
    )
    _b.scatter(_pos, _p_mean[_top], c=_colors, s=35, zorder=3)
    _b.set_xticks(_pos)
    _b.set_xticklabels([])
    _b.set_xlabel("top-20 ranked test cases")
    _b.set_ylabel(r"$\Pr(Y=1 \mid x)$ with 90% CI")
    _b.set_title(
        f"Top-20 hit rate: {y_cls_test[_top].sum()}/20  "
        f"(base rate {y_cls_test.mean():.0%})"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(X_cls, eta, idata_cls, model_cls, pm, pmb):
    # Variable importance: only X_0, X_1, X_2 generated the labels in
    # simulate_binary. The plot should rank those three at the top. We
    # reset X_data first because the OOS-prediction cell may have mutated
    # it.
    with model_cls:
        pm.set_data({"X_data": X_cls})
        _vi_cls = pmb.compute_variable_importance(idata_cls, eta, X_cls)
    pmb.plot_variable_importance(_vi_cls)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ordinal outcome: GSS 2022 job satisfaction

        BART composes with any PyMC likelihood. Here we model job
        satisfaction (`satjob`, four ordered levels) as an ordered probit
        whose latent score is BART. Predictors: age, four self-reported
        anxiety/stress scales, and one-hot indicators for sex, degree,
        race, and religion.

        The cutpoints are estimated jointly with the trees: the first
        cutpoint is fixed at zero for identifiability; the remaining two
        are constrained ordered.
        """
    )
    return


@app.cell
def _(Path, np, os, pl):
    def load_gss():
        override = os.environ.get("GSS_CSV")
        candidates = []
        if override:
            candidates.append(Path(override))
        candidates.extend(
            [
                Path.cwd() / "data" / "gss_2022.csv",
                Path.home() / "repos" / "Koenigsberg_Bayes" / "data" / "gss_2022.csv",
            ]
        )
        for p in candidates:
            if p.exists():
                return pl.read_csv(p)
        raise FileNotFoundError(
            "gss_2022.csv not found. Set GSS_CSV env var, place the file at "
            "./data/gss_2022.csv, or clone the Koenigsberg_Bayes repo."
        )

    _gss_raw = load_gss()
    _cont = ["age"]
    _ordinal = ["stress", "feelnerv", "worry", "anxiety", "finrela"]
    _categ = ["sex", "degree", "race", "relig"]
    _cols = ["satjob"] + _cont + _ordinal + _categ

    _df = _gss_raw.select(_cols).drop_nulls()
    y_ord = _df["satjob"].to_numpy().astype(int) - 1

    _X_parts = [_df[_cont + _ordinal].to_numpy().astype(float)]
    for _c in _categ:
        _dummies = _df[_c].to_dummies(drop_first=True).to_numpy().astype(float)
        _X_parts.append(_dummies)
    X_ord = np.concatenate(_X_parts, axis=1)

    f"n={len(y_ord)}, p={X_ord.shape[1]}, classes={np.bincount(y_ord).tolist()}"
    return X_ord, y_ord


@app.cell
def _(X_ord, np, pm, pmb, y_ord):
    with pm.Model() as model_sat:
        eta_sat = pmb.BART("eta", X=X_ord, Y=y_ord.astype(float), m=50)
        gamma_free = pm.Normal(
            "gamma_free",
            mu=np.array([1.0, 2.0]),
            sigma=1.0,
            size=2,
            transform=pm.distributions.transforms.ordered,
            initval=np.array([1.0, 2.0]),
        )
        cutpoints = pm.Deterministic(
            "cutpoints", pm.math.concatenate([[0.0], gamma_free])
        )
        pm.OrderedProbit(
            "y", eta=eta_sat, cutpoints=cutpoints, observed=y_ord, compute_p=False
        )
        idata_sat = pm.sample(
            draws=300,
            tune=300,
            chains=2,
            cores=1,
            random_seed=20260423,
            progressbar=False,
        )
    return eta_sat, idata_sat, model_sat


@app.cell
def _(az, idata_sat):
    az.summary(idata_sat, var_names=["gamma_free", "cutpoints"], round_to=3)
    return


@app.cell(hide_code=True)
def _(X_ord, eta_sat, idata_sat, model_sat, pmb):
    with model_sat:
        _vi_sat = pmb.compute_variable_importance(idata_sat, eta_sat, X_ord)
    pmb.plot_variable_importance(_vi_sat)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
