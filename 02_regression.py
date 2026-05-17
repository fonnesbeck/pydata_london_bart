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

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _():
    # pymc-bart's BART RV creates a multiprocessing.Manager(); force the
    # "fork" start method so this works under both marimo edit and bare
    # `python` script execution (Python 3.14 defaults to forkserver, which
    # re-imports the script and breaks without an `if __name__` guard).
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import pymc_bart as pmb

    RANDOM_SEED = 20260608
    rng = np.random.default_rng(RANDOM_SEED)
    return RANDOM_SEED, az, np, pl, plt, pm, pmb, rng


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
def _(RANDOM_SEED, X_fried, pm, pmb, y_fried):
    # The pymc-bart API: declare the BART random variable inside a PyMC
    # model alongside any other priors. Wrap X in pm.Data so we can swap it
    # later for out-of-sample predictions.
    with pm.Model() as model_fried:
        X_data = pm.Data("X_data", X_fried)
        mu_bart = pmb.BART("mu", X=X_data, Y=y_fried, m=200)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", mu=mu_bart, sigma=sigma, observed=y_fried, shape=mu_bart.shape)
        idata_fried = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RANDOM_SEED,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence diagnostics for a BART RV

    `mu` is a length-$n$ random variable, so `az.plot_trace` produces
    a wall of densities that is hard to read.
    `az.plot_convergence_dist` instead summarises across all `mu`
    components: the empirical CDF of effective sample size (left) and
    of $\hat R$ (right). The dashed lines are the rules-of-thumb
    (ESS $\ge 400$, $\hat R$ below a multiple-comparison-adjusted
    threshold).
    """)
    return


@app.cell(hide_code=True)
def _(az, idata_fried):
    az.plot_convergence_dist(idata_fried, var_names=["mu"])
    return


@app.cell
def _(az, idata_fried):
    _n_div = int(idata_fried.sample_stats["diverging"].sum())
    _summary = az.summary(idata_fried, var_names=["sigma"], round_to=3)
    f"divergences: {_n_div}   |   sigma: R-hat = {_summary['r_hat'].iloc[0]:.3f}, ESS = {_summary['ess_bulk'].iloc[0]:.0f}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior predictive check

    Sample draws of $y$ from the fitted posterior and overlay them on
    the observed data. If the model captures the data-generating
    process, the observed CDF (dark line) sits inside the cloud of
    posterior predictive CDFs (light lines). Misalignment here is
    the canonical signal of structural under-fit or
    over-/under-dispersion.
    """)
    return


@app.cell
def _(RANDOM_SEED, X_fried, idata_fried, model_fried, pm):
    with model_fried:
        pm.set_data({"X_data": X_fried})
        ppc_fried = pm.sample_posterior_predictive(
            idata_fried,
            var_names=["y"],
            sample_vars=["mu"],
            random_seed=RANDOM_SEED,
        )
    return (ppc_fried,)


@app.cell(hide_code=True)
def _(az, ppc_fried):
    az.plot_ppc_dist(ppc_fried, kind="ecdf", num_samples=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ECDF overlay tells us whether replicated draws cover the
    observed marginal. A complementary check is calibration via
    the **probability integral transform** (PIT): under a well
    calibrated model the PIT values $p(\tilde y_i \le y_i \mid y)$
    are uniform on $[0,1]$. `plot_ppc_pit` plots the Δ-ECDF of the
    PIT values; perfect calibration is the flat line at zero, and
    the shaded band is a simultaneous confidence envelope.
    """)
    return


@app.cell(hide_code=True)
def _(az, ppc_fried):
    az.plot_ppc_pit(ppc_fried)
    return


@app.cell
def _(RANDOM_SEED, X_fried_test, idata_fried, model_fried, pm):
    # Out-of-sample predictions: swap the X data, then sample posterior
    # predictive. pmb.BART evaluates the posterior trees at the new X.
    with model_fried:
        pm.set_data({"X_data": X_fried_test})
        pp_fried = pm.sample_posterior_predictive(
            idata_fried,
            var_names=["mu"],
            sample_vars=["mu"],
            predictions=True,
            random_seed=RANDOM_SEED,
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
    mo.md(r"""
    ### Choosing the number of trees with PSIS-LOO-CV

    `m` is BART's main knob. The Quiroga et al. paper recommends
    comparing fits at a few values of `m` using PSIS-LOO-CV
    [Vehtari et al., 2017] via `az.compare`. ELPD typically increases
    with `m` then plateaus; differences of $\le 4$ are not
    considered meaningful, so the smallest `m` whose ELPD is
    statistically indistinguishable from the largest is a defensible
    choice.

    Below we refit the Friedman model at $m \in \{10, 50, 200\}$
    (with `log_likelihood` enabled so LOO can use it) and compare.
    """)
    return


@app.cell
def _(RANDOM_SEED, X_fried, pm, pmb, y_fried):
    def fit_friedman(m):
        with pm.Model():
            X_data_m = pm.Data("X_data", X_fried)
            mu_m = pmb.BART("mu", X=X_data_m, Y=y_fried, m=m)
            sigma_m = pm.HalfNormal("sigma", 1.0)
            pm.Normal("y", mu=mu_m, sigma=sigma_m, observed=y_fried, shape=mu_m.shape)
            idata = pm.sample(
                draws=1000,
                tune=1000,
                chains=4,
                random_seed=RANDOM_SEED,
            )
            pm.compute_log_likelihood(idata)
        return idata

    return (fit_friedman,)


@app.cell
def _(fit_friedman):
    idata_fried_m10 = fit_friedman(10)
    return (idata_fried_m10,)


@app.cell
def _(fit_friedman):
    idata_fried_m50 = fit_friedman(50)
    return (idata_fried_m50,)


@app.cell
def _(fit_friedman):
    idata_fried_m200 = fit_friedman(200)
    return (idata_fried_m200,)


@app.cell(hide_code=True)
def _(az, idata_fried_m10, idata_fried_m200, idata_fried_m50, plt):
    cmp_m = az.compare(
        {
            "m=10": idata_fried_m10,
            "m=50": idata_fried_m50,
            "m=200": idata_fried_m200,
        }
    )
    _ax = az.plot_compare(cmp_m)
    plt.gcf().tight_layout()
    plt.gcf()
    return (cmp_m,)


@app.cell
def _(cmp_m):
    cmp_m
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The σ posterior tells the same story

    When BART can't represent the mean function $f(x)$ well, the
    leftover signal gets absorbed into the residual variance.
    Plotting $\sigma$'s posterior at each $m$ makes this concrete:
    too few trees → biased-high $\sigma$, too much "noise". The
    dashed line is the true $\sigma = 1$ that generated the data.
    """)
    return


@app.cell(hide_code=True)
def _(idata_fried_m10, idata_fried_m200, idata_fried_m50, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _label, _idata, _color in [
        ("m=10", idata_fried_m10, "#4c72b0"),
        ("m=50", idata_fried_m50, "#dd8452"),
        ("m=200", idata_fried_m200, "#55a868"),
    ]:
        _s = _idata.posterior["sigma"].values.ravel()
        _ax.hist(
            _s,
            bins=40,
            density=True,
            alpha=0.45,
            color=_color,
            label=f"{_label}  (mean={_s.mean():.2f})",
        )
    _ax.axvline(1.0, color="C3", ls="--", lw=1.2, label=r"true $\sigma = 1$")
    _ax.set_xlabel(r"$\sigma$")
    _ax.set_ylabel("posterior density")
    _ax.set_title(r"$\sigma$ shrinks toward the truth as $m$ grows")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real data: Formula 1 lap times

    The Friedman fit is calibration. Now a real-data application:
    predicting lap times at the 2024 British Grand Prix from in-race
    progress (tyre life, lap number, stint), weather (air/track
    temperature, humidity, wind), and tyre compound. Silverstone 2024
    had a wet phase, so the dataset includes INTERMEDIATE tyres
    alongside the dry SOFT/MEDIUM/HARD compounds.
    """)
    return


@app.cell
def _(pl):
    f1_df = pl.read_csv("data/f1_laps.csv")
    f1_df.shape
    return (f1_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aside: categorical splits with `OneHotSplitRule`

    Tyre compound is categorical with four levels (SOFT/MEDIUM/HARD/
    INTERMEDIATE at Silverstone 2024). The default `ContinuousSplitRule`
    treats the integer code as ordered, which is wrong: there is no
    natural ordering of compounds. `pmb.OneHotSplitRule` (Deshpande,
    2023) instead splits on `x == c` versus `x != c` for some level
    `c`, which is the right semantics for an unordered categorical.

    We pass `split_rules` as a length-`p` list with one rule per
    column: continuous for the seven numeric features, one-hot for
    the integer-coded compound.
    """)
    return


@app.cell
def _(RANDOM_SEED, f1_df, np):
    _num_cols = [
        "tyre_life",
        "lap_number",
        "stint",
        "air_temp",
        "track_temp",
        "humidity",
        "wind_speed",
    ]
    _compounds = sorted(f1_df["compound"].unique().to_list())
    _compound_code = {c: i for i, c in enumerate(_compounds)}
    f1_feature_names = list(_num_cols) + ["compound"]

    _X_num = f1_df.select(_num_cols).to_numpy().astype(float)
    _X_cmp = np.array(
        [_compound_code[c] for c in f1_df["compound"].to_numpy()], dtype=float
    )[:, None]
    _X = np.concatenate([_X_num, _X_cmp], axis=1)
    _y = f1_df["lap_time_s"].to_numpy().astype(float)
    n_num_f1 = len(_num_cols)

    _n_train = 500
    _n_test = 200
    _rng_f1 = np.random.default_rng(RANDOM_SEED)
    _perm = _rng_f1.permutation(_X.shape[0])
    X_train = _X[_perm[:_n_train]]
    y_train = _y[_perm[:_n_train]]
    X_test = _X[_perm[_n_train : _n_train + _n_test]]
    y_test = _y[_perm[_n_train : _n_train + _n_test]]
    return X_test, X_train, f1_feature_names, n_num_f1, y_test, y_train


@app.cell
def _(RANDOM_SEED, X_train, n_num_f1, pm, pmb, y_train):
    f1_split_rules = [pmb.ContinuousSplitRule] * n_num_f1 + [pmb.OneHotSplitRule]

    with pm.Model() as model_f1:
        X_data_f1 = pm.Data("X_data", X_train)
        mu_f1 = pmb.BART(
            "mu", X=X_data_f1, Y=y_train, m=100, split_rules=f1_split_rules
        )
        sigma_f1 = pm.HalfNormal("sigma", float(y_train.std()))
        pm.Normal("y", mu=mu_f1, sigma=sigma_f1, observed=y_train, shape=mu_f1.shape)
        idata_f1 = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RANDOM_SEED,
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


@app.cell(hide_code=True)
def _(az, idata_f1):
    az.plot_convergence_dist(idata_f1, var_names=["mu"])
    return


@app.cell
def _(az, idata_f1):
    _n_div = int(idata_f1.sample_stats["diverging"].sum())
    _summary = az.summary(idata_f1, var_names=["sigma"], round_to=3)
    f"divergences: {_n_div}   |   sigma: R-hat = {_summary['r_hat'].iloc[0]:.3f}, ESS = {_summary['ess_bulk'].iloc[0]:.0f}"
    return


@app.cell
def _(RANDOM_SEED, X_train, idata_f1, model_f1, pm):
    with model_f1:
        pm.set_data({"X_data": X_train})
        ppc_f1 = pm.sample_posterior_predictive(
            idata_f1,
            var_names=["y"],
            sample_vars=["mu"],
            random_seed=RANDOM_SEED,
        )
    return (ppc_f1,)


@app.cell(hide_code=True)
def _(az, ppc_f1):
    az.plot_ppc_dist(ppc_f1, kind="ecdf", num_samples=100)
    return


@app.cell(hide_code=True)
def _(az, ppc_f1):
    az.plot_ppc_pit(ppc_f1)
    return


@app.cell
def _(RANDOM_SEED, X_test, idata_f1, model_f1, pm):
    with model_f1:
        pm.set_data({"X_data": X_test})
        pp_f1 = pm.sample_posterior_predictive(
            idata_f1,
            var_names=["mu"],
            sample_vars=["mu"],
            predictions=True,
            random_seed=RANDOM_SEED,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnosing the F1 fit

    The PIT plot above says calibration is broken ($p \approx 0$):
    the posterior predictive is over-dispersed and slightly biased.
    The downstream heteroscedastic model will not rescue this on
    its own — when the mean function is structurally wrong,
    modelling $\sigma(x)$ just absorbs residual misfit.

    Two plausible levers:

    1. **Missing features.** The CSV has `driver`, `team`, and
       `position` columns we have not used. Driver-to-driver gaps
       in F1 are 0.3–0.8 s/lap and team-to-team gaps are several
       seconds; without those, BART cannot tell a back marker apart
       from a front runner.
    2. **Tree capacity.** Defaults $\alpha=0.95$, $\beta=2.0$ keep
       trees shallow (depth $\le 3$). Narrow regimes — slow
       out-laps, back-of-grid drivers — may need depth 5–6.

    Escalate one knob at a time and watch the PIT flatten.
    """)
    return


@app.cell
def _(RANDOM_SEED, f1_df, np, pl):
    # Reuse the same f1_df loaded above, but expand the feature set
    # to include driver, team, and position. Categorical columns are
    # encoded as integer category codes for pmb.OneHotSplitRule (same
    # pattern as the existing `compound` column).
    num_cols_full = [
        "tyre_life",
        "lap_number",
        "stint",
        "position",
        "air_temp",
        "track_temp",
        "humidity",
        "wind_speed",
    ]
    cat_cols_full = ["driver", "team", "compound"]
    n_numeric_full = len(num_cols_full)
    f1_feature_names_full = list(num_cols_full) + list(cat_cols_full)

    _X_num = f1_df.select(num_cols_full).to_numpy().astype(float)
    _X_cat = np.column_stack(
        [
            f1_df[c].cast(pl.Categorical).to_physical().to_numpy().astype(float)
            for c in cat_cols_full
        ]
    )
    _X_full = np.concatenate([_X_num, _X_cat], axis=1)
    _y_full = f1_df["lap_time_s"].to_numpy().astype(float)

    _rng_full = np.random.default_rng(RANDOM_SEED)
    _perm_full = _rng_full.permutation(_X_full.shape[0])
    X_train_full = _X_full[_perm_full[:500]]
    y_train_full = _y_full[_perm_full[:500]]
    return X_train_full, n_numeric_full, y_train_full


@app.cell
def _(RANDOM_SEED, n_numeric_full, pm, pmb):
    def fit_f1_homo(X, y, *, m=100, response="constant", alpha=0.95, beta=2.0):
        rules = [pmb.ContinuousSplitRule] * n_numeric_full + [pmb.OneHotSplitRule] * 3
        with pm.Model():
            _Xd = pm.Data("X_data", X)
            _mu = pmb.BART(
                "mu",
                X=_Xd,
                Y=y,
                m=m,
                response=response,
                alpha=alpha,
                beta=beta,
                split_rules=rules,
            )
            _sigma = pm.HalfNormal("sigma", float(y.std()))
            pm.Normal("y", mu=_mu, sigma=_sigma, observed=y, shape=_mu.shape)
            idata = pm.sample(
                draws=1000,
                tune=1000,
                chains=4,
                random_seed=RANDOM_SEED,
            )
            pm.sample_posterior_predictive(
                idata,
                var_names=["y"],
                sample_vars=["mu"],
                random_seed=RANDOM_SEED,
                extend_inferencedata=True,
            )
        return idata

    return (fit_f1_homo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit 1 — add `driver`, `team`, `position` (defaults otherwise)
    """)
    return


@app.cell
def _(X_train_full, fit_f1_homo, y_train_full):
    idata_f1_feat = fit_f1_homo(X_train_full, y_train_full)
    return (idata_f1_feat,)


@app.cell(hide_code=True)
def _(az, idata_f1_feat):
    az.plot_ppc_pit(idata_f1_feat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit 2 — features + `response='linear'`
    """)
    return


@app.cell
def _(X_train_full, fit_f1_homo, y_train_full):
    idata_f1_linear = fit_f1_homo(X_train_full, y_train_full, response="linear")
    return (idata_f1_linear,)


@app.cell(hide_code=True)
def _(az, idata_f1_linear):
    az.plot_ppc_pit(idata_f1_linear)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit 3 — features + linear leaves + $\beta=1.5$, $m=200$
    """)
    return


@app.cell
def _(X_train_full, fit_f1_homo, y_train_full):
    idata_f1_deep = fit_f1_homo(
        X_train_full, y_train_full, response="linear", beta=1.5, m=200
    )
    return (idata_f1_deep,)


@app.cell(hide_code=True)
def _(az, idata_f1_deep):
    az.plot_ppc_pit(idata_f1_deep)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit 4 — Student-T likelihood (linear leaves, $\beta=1.5$, $m=200$)

    σ̂ has come down meaningfully with more capacity, but the PIT
    shape barely moves. That is the signature of a misspecified
    *likelihood*, not a misspecified mean. F1 residuals retain a
    right tail (the SC / lift / dirty-air laps that survive the
    0.9–1.3 median filter in `pull_f1_laps.py`), and a symmetric
    Gaussian likelihood cannot represent them at any $(\mu, \sigma)$
    setting. Swap `pm.Normal` for `pm.StudentT` and let $\nu$ learn
    the tail heaviness.
    """)
    return


@app.cell
def _(RANDOM_SEED, X_train_full, n_numeric_full, pm, pmb, y_train_full):
    _rules_t = [pmb.ContinuousSplitRule] * n_numeric_full + [pmb.OneHotSplitRule] * 3
    with pm.Model() as model_f1_t:
        _Xd = pm.Data("X_data", X_train_full)
        _mu = pmb.BART(
            "mu",
            X=_Xd,
            Y=y_train_full,
            m=200,
            response="linear",
            alpha=0.95,
            beta=1.5,
            split_rules=_rules_t,
        )
        _sigma = pm.HalfNormal("sigma", float(y_train_full.std()))
        _nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
        pm.StudentT(
            "y",
            nu=_nu,
            mu=_mu,
            sigma=_sigma,
            observed=y_train_full,
            shape=_mu.shape,
        )
        idata_f1_t = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RANDOM_SEED,
        )
        pm.sample_posterior_predictive(
            idata_f1_t,
            var_names=["y"],
            sample_vars=["mu"],
            random_seed=RANDOM_SEED,
            extend_inferencedata=True,
        )
    return idata_f1_t, model_f1_t


@app.cell(hide_code=True)
def _(az, idata_f1_t):
    az.plot_ppc_pit(idata_f1_t)
    return


@app.cell
def _(RANDOM_SEED, X_test, idata_f1_t, model_f1_t, pm):
    with model_f1_t:
        pm.set_data({"X_data": X_test})
        pp_f1_t = pm.sample_posterior_predictive(
            idata_f1_t,
            var_names=["mu"],
            sample_vars=["mu"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return


@app.cell
def _(idata_f1_t, np, plt, y_test):
    _mu_pred = idata_f1_t.predictions["mu"].stack(sample=("chain", "draw")).values
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
def _(az, idata_f1_deep, idata_f1_feat, idata_f1_linear, idata_f1_t, pl):
    def _row(name, idata, with_nu=False):
        _sig = float(idata.posterior["sigma"].mean().item())
        _ndiv = int(idata.sample_stats["diverging"].sum().item())
        _summary = az.summary(idata, var_names=["mu"], round_to=3)
        _nu = round(float(idata.posterior["nu"].mean().item()), 1) if with_nu else None
        return {
            "variant": name,
            "sigma_hat": round(_sig, 3),
            "nu_hat": _nu,
            "divergences": _ndiv,
            "rhat_max": round(float(_summary["r_hat"].max()), 3),
            "ess_min": int(_summary["ess_bulk"].min()),
        }

    pl.DataFrame(
        [
            _row("features only", idata_f1_feat),
            _row("+ linear leaves", idata_f1_linear),
            _row("+ linear, β=1.5, m=200", idata_f1_deep),
            _row("+ Student-T", idata_f1_t, with_nu=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Feature audit usually beats hyperparameter tuning: adding
    `driver` and `team` does most of the work. `response='linear'`
    extends the leaf model into the slow-lap tail, where constant
    leaves struggle to extrapolate. Deepening the trees
    ($\beta=1.5$, $m=200$) polishes residuals further but watch
    for overfitting on $n=500$.

    The heteroscedastic BART below deliberately keeps the original
    feature set as the teaching baseline. The lesson cuts both
    ways: when the mean is under-specified, $\sigma(x)$ absorbs
    the misfit. Re-running it on `X_train_full` is left as an
    exercise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heteroscedastic BART: modelling $\sigma(x)$

    So far we have used BART to model the mean and assumed a single
    scalar $\sigma$. But F1 lap-time variance plausibly depends on
    covariates: stint phase, wet/dry conditions, tyre compound.
    Quiroga et al. §4.4 show that since `pmb.BART` is a primitive
    random variable, it can be wrapped around any parameter of any
    likelihood — including the standard deviation.

    We use **two BART random variables in one model**: one for
    $\mu(x)$ and one for $\log \sigma(x)$. The trick is to pass each
    BART a `Y` argument at the right scale so its leaf values
    initialise sensibly:

    - `mu` gets `Y=y_train` (sum of trees starts near $\bar y$).
    - `log_sigma` gets `Y=\log(|y - \bar y| + 0.5)`, so the sum of
      trees starts near $\log(\text{typical SD})$ — not
      $\log(\bar y)$, which would give an astronomical $\sigma$.

    The fully non-parametric `size=2` alternative from the paper
    (Code Block 6) collapses both outputs to the same initialisation
    and is sensitive to tuning length; two separate BARTs avoid that.
    """)
    return


@app.cell
def _(RANDOM_SEED, X_train, n_num_f1, np, pm, pmb, y_train):
    f1_split_rules_het = [pmb.ContinuousSplitRule] * n_num_f1 + [pmb.OneHotSplitRule]
    # Y_log_dev: pass log of typical |y - mean(y)| as Y to the σ-BART so
    # leaf initialisation lands near log(typical SD), not log(mean y).
    _y_log_dev = np.log(np.abs(y_train - y_train.mean()) + 0.5)
    with pm.Model() as model_het:
        X_data_het = pm.Data("X_data", X_train)
        mu_het = pmb.BART(
            "mu",
            X=X_data_het,
            Y=y_train,
            m=200,
            split_rules=f1_split_rules_het,
        )
        log_sigma = pmb.BART(
            "log_sigma",
            X=X_data_het,
            Y=_y_log_dev,
            m=50,
            split_rules=f1_split_rules_het,
        )
        sigma_het = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        pm.Normal("y", mu=mu_het, sigma=sigma_het, observed=y_train)
        idata_het = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=RANDOM_SEED,
        )
    return (idata_het,)


@app.cell(hide_code=True)
def _(idata_het, np, plt, y_train):
    _mu = idata_het.posterior["mu"].stack(sample=("chain", "draw")).values
    _sd = idata_het.posterior["sigma"].stack(sample=("chain", "draw")).values
    _mu_mean = _mu.mean(axis=1)
    _sd_mean = _sd.mean(axis=1)

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.0))
    _order = np.argsort(_mu_mean)
    _a.scatter(_mu_mean[_order], y_train[_order], s=12, alpha=0.55, color="#4c72b0")
    _lim = (
        min(_mu_mean.min(), y_train.min()) - 0.5,
        max(_mu_mean.max(), y_train.max()) + 0.5,
    )
    _a.plot(_lim, _lim, "--", color="C3", lw=1)
    _a.set_xlim(_lim)
    _a.set_ylim(_lim)
    _a.set_xlabel(r"posterior mean $\hat \mu(x)$")
    _a.set_ylabel("observed lap time (s)")
    _a.set_title("Heteroscedastic BART: fit")

    _b.scatter(_mu_mean[_order], _sd_mean[_order], s=12, alpha=0.6, color="#c44e52")
    _b.set_xlabel(r"posterior mean $\hat \mu(x)$")
    _b.set_ylabel(r"posterior mean $\hat \sigma(x)$ (s)")
    _b.set_title(r"$\sigma(x)$ varies with $\mu(x)$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(az, idata_het):
    az.plot_convergence_dist(idata_het, var_names=["mu"])
    return


@app.cell(hide_code=True)
def _(az, idata_het):
    az.plot_convergence_dist(idata_het, var_names=["log_sigma"])
    return


@app.cell
def _(idata_het):
    _stats = idata_het.sample_stats
    (
        f"divergences: {int(_stats['diverging'].sum())}"
        if "diverging" in _stats
        else "PGBART-only model (no HMC step) — divergence diagnostic not applicable"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
