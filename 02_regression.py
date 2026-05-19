import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BART for regression

    We use `pymc-bart`, the production library that wraps the BART
    algorithm with PyMC's sampler and posterior tooling, and apply it
    to a real-world regression: **predicting lap times across five 2024
    Formula 1 Grands Prix** (Bahrain, Monaco, Spain, Britain, Monza).

    The five races span the lap-time range of the calendar (Monaco's
    ~75s street laps to Monza's ~85s low-downforce blasts; Britain and
    Spain in the middle) and include dry, hot, and wet phases. An 80/20
    random split holds out roughly 1000 laps for testing.

    For each fit we plot the posterior predictive HDI band, check
    calibration with the PIT, rank variable importance, and visualise
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
    return RANDOM_SEED, az, np, pl, plt, pm, pmb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The dataset

    `data/f1_laps.csv` was assembled by `scripts/pull_f1_laps.py`,
    which pulls race laps via FastF1 for five 2024 Grands Prix and
    filters each race to the 0.9-1.3 median-lap-time band (drops pit
    in/out, deleted laps, safety-car phases). Each row is one clean
    racing lap.

    Predictors: in-race progress (tyre life, lap number, stint),
    running position, weather (air/track temperature, humidity, wind),
    tyre compound, and the `venue` categorical naming the race.
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
    ### Aside: categorical splits with `OneHotSplitRule` and `SubsetSplitRule`

    Two of our features are unordered categoricals. The default
    `ContinuousSplitRule` would treat their integer codes as ordered,
    which is wrong. Two alternatives:

    - **`pmb.OneHotSplitRule`** (Deshpande, 2023) splits on `x == c`
      versus `x != c` for some level `c`. Right semantics for any
      unordered categorical; the tree chains $K-1$ binary splits to
      recover a $K$-level partition.
    - **`pmb.SubsetSplitRule`** splits on $x \in S$ for an arbitrary
      subset $S \subset \{0, \ldots, K-1\}$ in a single node. Cheaper
      and more expressive when $K$ is small.

    For tyre compound (four levels) we use one-hot; for `venue` (five
    levels) we use subset. We pass `split_rules` as a length-$p$ list,
    one rule per column.
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
    _venues = sorted(f1_df["venue"].unique().to_list())
    _venue_code = {v: i for i, v in enumerate(_venues)}
    f1_feature_names = list(_num_cols) + ["compound", "venue"]

    _X_num = f1_df.select(_num_cols).to_numpy().astype(float)
    _X_cmp = np.array(
        [_compound_code[c] for c in f1_df["compound"].to_numpy()], dtype=float
    )[:, None]
    _X_ven = np.array([_venue_code[v] for v in f1_df["venue"].to_numpy()], dtype=float)[
        :, None
    ]
    _X = np.concatenate([_X_num, _X_cmp, _X_ven], axis=1)
    _y = f1_df["lap_time_s"].to_numpy().astype(float)
    n_num_f1 = len(_num_cols)

    # Random 80/20 split across all 5 venues. Every venue appears in both
    # train and test, so the venue feature has no out-of-distribution issue
    # at prediction time.
    _rng_split = np.random.default_rng(RANDOM_SEED)
    _perm = _rng_split.permutation(_X.shape[0])
    _n_train = int(0.8 * _X.shape[0])
    train_idx = _perm[:_n_train]
    test_idx = _perm[_n_train:]
    X_train = _X[train_idx]
    y_train = _y[train_idx]
    X_test = _X[test_idx]
    y_test = _y[test_idx]

    f"train: {X_train.shape[0]} laps | test: {X_test.shape[0]} laps | venues: {len(_venues)}"
    return (
        X_test,
        X_train,
        f1_feature_names,
        n_num_f1,
        test_idx,
        train_idx,
        y_test,
        y_train,
    )


@app.cell
def _(RANDOM_SEED, X_train, n_num_f1, pm, pmb, y_train):
    f1_split_rules = (
        [pmb.ContinuousSplitRule] * n_num_f1
        + [pmb.OneHotSplitRule]
        + [pmb.SubsetSplitRule]
    )

    with pm.Model() as model_f1:
        X_data_f1 = pm.Data("X_data", X_train)
        mu_f1 = pmb.BART(
            "mu", X=X_data_f1, Y=y_train, m=100, split_rules=f1_split_rules
        )
        sigma_f1 = pm.HalfNormal("sigma", float(y_train.std()))
        pm.Normal("y", mu=mu_f1, sigma=sigma_f1, observed=y_train, shape=mu_f1.shape)
        idata_f1 = pm.sample(random_seed=RANDOM_SEED)
    return f1_split_rules, idata_f1, model_f1, mu_f1


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
    _resid = _f_mean - y_test
    _cov = ((_f_lo <= y_test) & (y_test <= _f_hi)).mean()

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.5))
    _a.scatter(y_test, _f_mean, s=10, alpha=0.35, color="#4c72b0", edgecolor="none")
    _lim = (y_test.min() - 0.5, y_test.max() + 0.5)
    _a.plot(_lim, _lim, "--", color="C3", lw=1)
    _a.set_xlim(_lim)
    _a.set_ylim(_lim)
    _a.set_xlabel("observed lap time (s)")
    _a.set_ylabel(r"posterior mean $\hat f(x)$")
    _a.set_title(f"F1 baseline OOS: 90% HDI coverage = {_cov:.0%}")

    _b.hist(_resid, bins=40, color="#4c72b0", alpha=0.8, edgecolor="white")
    _b.axvline(0.0, color="C3", ls="--", lw=1)
    _b.set_xlabel(r"residual $\hat f(x) - y$  (s)")
    _b.set_ylabel("count")
    _b.set_title(
        f"residual distribution  (mean={_resid.mean():.2f}, sd={_resid.std():.2f})"
    )
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
    its own - when the mean function is structurally wrong,
    modelling $\sigma(x)$ just absorbs residual misfit.

    During development we escalated four levers one at a time and
    watched the PIT flatten:

    1. **Add `driver`, `team`, `position`.** Driver-to-driver gaps
       are 0.3-0.8 s/lap, team gaps are several seconds. Without
       those, BART cannot tell a back marker apart from a front
       runner. Single biggest improvement.
    2. **`response="linear"` leaves.** Each leaf carries a small
       linear fit instead of a constant. Helps extrapolation into
       the slow-lap tail.
    3. **$\beta = 1.5$, $m = 200$.** The depth prior allows deeper
       trees ($\beta$ smaller -> less penalty for depth) and the
       larger ensemble gives more room to absorb structure.
    4. **`pm.StudentT` likelihood with learned $\nu$.** Even after
       levers 1-3, the PIT shape barely moved. The remaining
       miscalibration is in the *likelihood*, not the mean. F1
       residuals retain a right tail (the SC / lift / dirty-air
       laps that survive the 0.9-1.3 median filter in
       `pull_f1_laps.py`); a symmetric Gaussian cannot represent
       them at any $(\mu, \sigma)$ setting.

    Below we show the fully-escalated fit (all four levers applied
    at once).
    """)
    return


@app.cell
def _(f1_df, np, pl, test_idx, train_idx):
    # Expanded feature set: adds driver, team, position to the baseline.
    # Categorical columns are integer-coded via polars Categorical -> physical
    # so OneHotSplitRule and SubsetSplitRule can split on them.
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
    cat_cols_full = ["driver", "team", "compound", "venue"]
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

    # Reuse the train/test indices from the baseline data prep so the
    # baseline and Student-T fits are trained / tested on the same rows.
    X_train_full = _X_full[train_idx]
    y_train_full = _y_full[train_idx]
    X_test_full = _X_full[test_idx]
    y_test_full = _y_full[test_idx]
    return X_test_full, X_train_full, n_numeric_full, y_test_full, y_train_full


@app.cell
def _(RANDOM_SEED, X_train_full, n_numeric_full, pm, pmb, y_train_full):
    # Final escalation: expanded feature set + linear leaves + beta=1.5
    # + m=200 + StudentT(nu) likelihood. driver/team/compound stay on
    # OneHotSplitRule; venue uses SubsetSplitRule (multi-level unordered).
    _rules_t = (
        [pmb.ContinuousSplitRule] * n_numeric_full
        + [pmb.OneHotSplitRule] * 3
        + [pmb.SubsetSplitRule]
    )
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
        idata_f1_t = pm.sample(random_seed=RANDOM_SEED)
    return idata_f1_t, model_f1_t


@app.cell
def _(RANDOM_SEED, idata_f1_t, model_f1_t, pm, y_train_full):
    # PPC against the training data is fast; do it live so the PIT plot
    # below reflects the same idata we just loaded.
    with model_f1_t:
        ppc_f1_t = pm.sample_posterior_predictive(
            idata_f1_t,
            var_names=["y"],
            sample_vars=["mu"],
            random_seed=RANDOM_SEED,
            extend_inferencedata=True,
        )
    _ = y_train_full
    return (ppc_f1_t,)


@app.cell(hide_code=True)
def _(az, ppc_f1_t):
    az.plot_ppc_pit(ppc_f1_t)
    return


@app.cell
def _(RANDOM_SEED, X_test_full, idata_f1_t, model_f1_t, pm):
    # OOS prediction uses the held-out British GP. X_test_full has 12
    # features (matching the Student-T model's expanded feature set);
    # X_test from the baseline has 9 features and would shape-mismatch
    # against the trained trees.
    with model_f1_t:
        pm.set_data({"X_data": X_test_full})
        pp_f1_t = pm.sample_posterior_predictive(
            idata_f1_t,
            var_names=["mu"],
            sample_vars=["mu"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_f1_t,)


@app.cell
def _(np, plt, pp_f1_t, y_test_full):
    _mu_pred = pp_f1_t.predictions["mu"].stack(sample=("chain", "draw")).values
    _f_mean = _mu_pred.mean(axis=1)
    _f_lo, _f_hi = np.quantile(_mu_pred, [0.05, 0.95], axis=1)
    _resid = _f_mean - y_test_full
    _cov = ((_f_lo <= y_test_full) & (y_test_full <= _f_hi)).mean()

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.5))
    _a.scatter(
        y_test_full, _f_mean, s=10, alpha=0.35, color="#4c72b0", edgecolor="none"
    )
    _lim = (y_test_full.min() - 0.5, y_test_full.max() + 0.5)
    _a.plot(_lim, _lim, "--", color="C3", lw=1)
    _a.set_xlim(_lim)
    _a.set_ylim(_lim)
    _a.set_xlabel("observed lap time (s)")
    _a.set_ylabel(r"posterior mean $\hat f(x)$")
    _a.set_title(f"F1 (Student-T) OOS: 90% HDI coverage = {_cov:.0%}")

    _b.hist(_resid, bins=40, color="#4c72b0", alpha=0.8, edgecolor="white")
    _b.axvline(0.0, color="C3", ls="--", lw=1)
    _b.set_xlabel(r"residual $\hat f(x) - y$  (s)")
    _b.set_ylabel("count")
    _b.set_title(
        f"residual distribution  (mean={_resid.mean():.2f}, sd={_resid.std():.2f})"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(az, idata_f1, idata_f1_t, pl):
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
            _row("baseline (8 features, Normal)", idata_f1),
            _row("+ features + linear + β=1.5 + Student-T", idata_f1_t, with_nu=True),
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
    ($\beta=1.5$, $m=200$) polishes residuals further; with
    $n \approx 4{,}150$ training laps across the 5 venues the model
    has enough signal to support the deeper ensemble.

    The heteroscedastic BART below uses the **baseline** feature set
    (7 numeric + `compound` + `venue`) as the teaching contrast. The
    lesson cuts both ways: when the mean is under-specified,
    $\sigma(x)$ absorbs the misfit. Re-running it on the expanded
    feature set is left as an exercise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing the number of trees with PSIS-LOO-CV

    `m` is BART's main knob: how many trees vote on the prediction.
    The Quiroga et al. paper recommends comparing fits at a few
    values of `m` using PSIS-LOO-CV [Vehtari et al., 2017] via
    `az.compare`. ELPD typically increases with `m` then plateaus;
    differences of $\le 4$ are not considered meaningful, so the
    smallest `m` whose ELPD is statistically indistinguishable from
    the largest is a defensible choice.

    Below we refit the **baseline** F1 model (Normal likelihood,
    9 features) at $m \in \{10, 50, 200\}$. Keeping the likelihood
    and feature set fixed isolates the `m` lever from the Student-T
    escalation above; the LOO numbers are directly comparable.
    """)
    return


@app.cell(hide_code=True)
def _(RANDOM_SEED, X_train, f1_split_rules, pm, pmb, y_train):
    def fit_f1_m(m):
        with pm.Model() as model_m:
            X_data_m = pm.Data("X_data", X_train)
            mu_m = pmb.BART(
                "mu", X=X_data_m, Y=y_train, m=m, split_rules=f1_split_rules
            )
            sigma_m = pm.HalfNormal("sigma", float(y_train.std()))
            pm.Normal("y", mu=mu_m, sigma=sigma_m, observed=y_train, shape=mu_m.shape)
            idata = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(idata)
        return idata

    return (fit_f1_m,)


@app.cell(hide_code=True)
def _(fit_f1_m):
    idata_f1_m10 = fit_f1_m(10)
    return (idata_f1_m10,)


@app.cell(hide_code=True)
def _(fit_f1_m):
    idata_f1_m50 = fit_f1_m(50)
    return (idata_f1_m50,)


@app.cell(hide_code=True)
def _(fit_f1_m):
    idata_f1_m200 = fit_f1_m(200)
    return (idata_f1_m200,)


@app.cell(hide_code=True)
def _(az, idata_f1_m10, idata_f1_m200, idata_f1_m50, plt):
    cmp_m = az.compare(
        {
            "m=10": idata_f1_m10,
            "m=50": idata_f1_m50,
            "m=200": idata_f1_m200,
        }
    )
    _ax = az.plot_compare(cmp_m)
    plt.gcf().tight_layout()
    plt.gcf()
    return (cmp_m,)


@app.cell(hide_code=True)
def _(cmp_m):
    cmp_m
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The $\sigma$ posterior tells the same story

    When BART can't represent the mean function $f(x)$ well, the
    leftover signal gets absorbed into the residual variance.
    Plotting $\sigma$'s posterior at each `m` makes this concrete:
    too few trees -> biased-high $\sigma$, too much "noise" the
    model attributes to randomness. As `m` grows the histogram
    shifts left toward the irreducible lap-to-lap variation.
    """)
    return


@app.cell(hide_code=True)
def _(idata_f1_m10, idata_f1_m200, idata_f1_m50, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _label, _idata, _color in [
        ("m=10", idata_f1_m10, "#4c72b0"),
        ("m=50", idata_f1_m50, "#dd8452"),
        ("m=200", idata_f1_m200, "#55a868"),
    ]:
        _s = _idata.posterior["sigma"].values.ravel()
        _ax.hist(
            _s,
            bins=40,
            density=True,
            alpha=0.45,
            color=_color,
            label=f"{_label}  (mean={_s.mean():.2f}s)",
        )
    _ax.set_xlabel(r"$\sigma$ (s)")
    _ax.set_ylabel("posterior density")
    _ax.set_title(r"$\sigma$ shrinks toward the data noise floor as $m$ grows")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
