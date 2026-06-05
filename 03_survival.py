import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BART for survival analysis: Tommy John surgery recovery

    Tommy John surgery, ulnar collateral ligament reconstruction, is
    the defining career intervention for baseball pitchers. The public
    record of who got it, when, and how long they took to return makes
    a textbook survival dataset. We use Jon Roegele's
    community-maintained list (`data/tommy_john.parquet`), restricted
    to surgeries between 1990 and 2022 across every professional level
    (MLB, AAA, AA, A, College, High School). Roegele's descriptive
    table shows recovery time roughly flat from age 18 to 27, then
    rising through the early 30s; rehab protocols changed around 2005,
    shortening average return time; the top eight surgeons account for
    a third of all surgeries.

    **Survival basics.** The event is returning to play. A censored player is
    observed for some amount of time without a recorded return; they may still
    be at risk after our observation window ends. The **at-risk set** at month
    $t$ is everyone who has not yet returned or been censored before that
    month. The **hazard** is the chance of returning during month $t$ given
    that the player is still at risk, and the **survival curve** is the
    probability of still not having returned by month $t$.

    We model **discrete-time hazards**: at each integer month $t$
    post-surgery the return event occurs with probability
    $$h(t \mid x) = 1 - \exp\!\bigl(-\exp(g(t, x))\bigr),$$
    where $g$ is a sum of trees and the link is complementary log-log
    (cloglog). Each subject contributes one row per month they remain at risk;
    the outcome is "did they return this month?". That is exactly the binary
    classification setup from the classification notebook, applied to expanded
    person-time data. Survival follows from the hazards:
    $$S(t \mid x) = \prod_{s \le t} \bigl(1 - h(s \mid x)\bigr).$$

    BART gives a flexible alternative to a proportional-hazards analysis
    when the shape of the hazard is not known in advance.
    """)
    return


@app.cell(hide_code=True)
def _():
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import pymc_bart as pmb

    RANDOM_SEED = 20260608
    return RANDOM_SEED, az, np, pl, plt, pm, pmb


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
            ## Tommy John clinical context

            Tommy John surgery rebuilds the elbow's **ulnar collateral ligament
            (UCL)**, the medial ligament that resists the valgus stress of
            high-velocity throwing. When the UCL can no longer stabilize the
            joint, surgeons typically harvest a tendon graft, drill tunnels in
            the humerus and ulna, and thread the graft through those tunnels to
            replace the torn ligament. 
            """),
            mo.md(r"""
            The operation is named for Tommy John,
            the pitcher who underwent Dr. Frank Jobe's first successful
            professional-baseball UCL reconstruction in 1974; modern
            return-to-competitive-throwing rehab still commonly runs 12--18
            months.
            """),
            mo.image(
                "images/ucl.webp",
                alt="Diagram of elbow medial collateral ligament bundles and Tommy John UCL reconstruction with tendon graft threaded through humerus and ulna tunnels",
                width="100%",
                rounded=True,
                caption="The UCL spans the medial elbow between humerus and ulna; reconstruction replaces the damaged ligament with a tendon graft routed through bone tunnels.",
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The real Tommy John data

    We apply the discrete-time hazard BART, alongside a cloglog GLM
    comparator, to the Roegele Tommy John surgery list.

    We keep all professional levels and all positions (not just MLB
    pitchers), so BART sees the full range of recovery trajectories
    Roegele compiled; the broader population is what makes Roegele's
    descriptive table show the nonlinear age effect, and the MLB-only
    subset is too narrow to recover it. Surgeries are restricted to
    1990--2022, a handful of `throws` records (`R*`, `P`, `L/R`) are
    dropped as data-entry artifacts, and a 36-month administrative
    horizon is imposed: anyone still not returned by then is censored
    at $t = 36$.

    The data-subsample control below defaults to 500 subjects, which keeps the
    person-month expansion to roughly 12k rows. The fixed seed makes the
    displayed subset reproducible.
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    HORIZON = 36.0
    tj_all = (
        pl.read_parquet("data/tommy_john.parquet")
        .filter(
            (pl.col("surgery_year") >= 1990)
            & (pl.col("surgery_year") <= 2022)
            & pl.col("time_months").is_not_null()
            & pl.col("age").is_not_null()
            & pl.col("surgeon_group").is_not_null()
            & pl.col("revision").is_not_null()
            & (pl.col("time_months") > 0)
            & pl.col("throws").is_in(["R", "L"])
        )
        .with_columns(
            event=pl.when(pl.col("time_months") > HORIZON)
            .then(0)
            .otherwise(pl.col("event"))
            .cast(pl.Int8),
            time_months=pl.when(pl.col("time_months") > HORIZON)
            .then(HORIZON)
            .otherwise(pl.col("time_months")),
        )
    )
    f"{tj_all.height} subjects pass the 1990-2022 filters"
    return HORIZON, tj_all


@app.cell(hide_code=True)
def _(mo, tj_all):
    _subsample_options = {
        f"Full ({tj_all.height} subjects)": tj_all.height,
        "1000 subjects": 1000,
        "500 subjects": 500,
        "250 subjects": 250,
    }
    subsample = mo.ui.radio(
        options=_subsample_options,
        value="500 subjects",
        label="**Data subsample** &mdash; fewer subjects = faster fits for iteration",
    )
    subsample
    return (subsample,)


@app.cell(hide_code=True)
def _(HORIZON, RANDOM_SEED, subsample, tj_all):
    tj_df = tj_all.sample(n=int(subsample.value), seed=RANDOM_SEED, shuffle=True)
    f"{tj_df.height} subjects, {int(tj_df['event'].sum())} returns, {tj_df.height - int(tj_df['event'].sum())} censored at {HORIZON:.0f}-month horizon"
    return (tj_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > ### Censoring vs competing events
    >
    > The discrete-time hazard model below treats every row with
    `event=0` as standard right-censoring: the subject was still at
    risk when our observation ended, and could in principle have
    returned later. That's the right interpretation for someone who
    is still in rehab as of the data pull.
    >
    > It's the *wrong* interpretation for a pitcher whose career ended
    without a return. Roegele's data dictionary classifies 64% of
    non-returns as exactly this — *competing events* (`active=0`
    with no return), not right-censoring. Treating career-enders as
    "still at risk" implicitly assumes they would have come back
    eventually, which biases the modelled survival curves upward.
    >
    > The 36-month administrative horizon helps the worst cases (anyone
    that has been lost track of) but does not address
    competing events inside the horizon.
    >
    > The best solution to this issue is a
    **competing-risks model**. Sparapani et al. (2020, *SMMR* 29:57-77)
    extend BART to subdistribution hazards for exactly this setup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expanding to person-month format

    The model sees a long-format expansion of the subject table: each
    subject contributes one row per integer month they remain at risk,
    ending in a final row where they either return (`y=1`) or are
    censored (`y=0`). The discrete time index `t` goes in column 0 of
    `X`, so BART can split on it and learn the hazard's time shape
    jointly with the covariate effects rather than imposing a
    parametric baseline.

    Categorical columns are integer-coded against sorted level lists
    that we keep around, so prediction profiles later encode levels by
    name (`surgeon_levels.index(...)`) rather than hardcoding codes.
    Using the same typed split-rule assignment introduced in the regression
    notebook, `split_rules` assigns each column the right tree geometry:
    continuous splits for
    `t`/`age`/`year`/`doy`, one-hot for binary `throws` and `revision`, and
    subset splits for the 10-level surgeon group. We use `max(1, ceil(time))`
    in the expansion so the rare same-day censor still contributes one
    at-risk row.
    """)
    return


@app.cell
def _(np, pmb, tj_df):
    surgeon_levels = sorted(tj_df["surgeon_group"].unique().to_list())
    throws_levels = sorted(tj_df["throws"].unique().to_list())
    surgeon_code = {s: i for i, s in enumerate(surgeon_levels)}
    throws_code = {th: i for i, th in enumerate(throws_levels)}

    _rows = []
    for _r in tj_df.iter_rows(named=True):
        _n_periods = max(1, int(np.ceil(_r["time_months"])))
        for _t in range(1, _n_periods + 1):
            _rows.append(
                (
                    _t,
                    _r["age"],
                    _r["surgery_year"],
                    _r["surgery_doy_frac"],
                    throws_code[_r["throws"]],
                    _r["revision"],
                    surgeon_code[_r["surgeon_group"]],
                    _r["event"] if _t == _n_periods else 0,
                )
            )
    _arr = np.array(_rows, dtype=float)
    surv_X = _arr[:, :7]
    surv_y = _arr[:, 7].astype(int)

    feature_names = ["t", "age", "year", "doy", "throws", "rev", "surgeon"]
    split_rules = (
        [pmb.ContinuousSplitRule] * 4
        + [pmb.OneHotSplitRule] * 2
        + [pmb.SubsetSplitRule]
    )
    f"{surv_X.shape[0]} person-month rows, {int(surv_y.sum())} return events"
    return (
        feature_names,
        split_rules,
        surgeon_levels,
        surv_X,
        surv_y,
        throws_levels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A classical comparator in the same framework

    Before BART, the standard approach to data like this would be a
    generalized linear model on the person-month expansion. The same
    binary indicator (did the pitcher return this month?) is regressed
    on the same covariates, but instead of a sum of trees the linear
    predictor is a *linear* combination of covariate effects plus a
    flexible baseline hazard.

    With a **complementary log-log** link,
    $$h(t \mid x) \;=\; 1 - \exp\!\bigl(-\exp(\alpha_t + x^\top \beta)\bigr),$$
    the interval cumulative hazard is
    $$\Lambda(t \mid x) = -\log\{1 - h(t \mid x)\} = \exp(\alpha_t + x^\top \beta).$$
    The ratio $\Lambda(t \mid x_1) / \Lambda(t \mid x_2)$ is exactly
    $\exp\{\beta^\top (x_1 - x_2)\}$, constant over $t$. This is the
    discrete-time form of the Cox proportional hazards model. The raw
    event-probability ratio $h(t \mid x_1) / h(t \mid x_2)$ is only
    approximately the same when monthly hazards are small.

    We give the baseline log-interval-hazard $\alpha_t$ twelve degrees of
    freedom (one log-hazard per quarter, $t \in \{1\text{-}3, 4\text{-}6,
    \ldots, 34\text{-}36\}$), which is plenty of flexibility on the time
    axis. Any remaining disagreement with BART will then be honestly about
    the covariate effects, not the baseline.

    BART uses the same cloglog link as this GLM, so the classical-vs-BART
    contrast lives on a single axis: the function class of the linear
    predictor. Linear additive (GLM) vs. tree sum (BART). If $g(t, x)$
    were linear in this notebook's BART model, strict proportional hazards
    would hold there too; any time-varying hazard ratio we recover is a
    consequence of the tree-sum function class, not the link.

    Concretely, the design matrix below standardizes the continuous
    covariates, passes the binary indicators through, and dummy-codes
    the surgeon category with the largest group (Dr. James Andrews) as
    the reference. The quarter index `q_idx` is carried separately so
    the baseline log-hazard $\alpha_q$ can be a 12-vector indexed at
    row level rather than a 12-column dummy block in the matrix. Each
    $\beta_j$ is then interpretable as a log hazard ratio that applies
    at every $t$ by construction. Because this model contains no BART
    variable, `pm.sample` uses NUTS.
    """)
    return


@app.cell
def _(np, surgeon_levels, surv_X):
    _t_col = surv_X[:, 0].astype(int)
    q_idx = ((_t_col - 1) // 3).astype(int)

    _age = surv_X[:, 1]
    _year = surv_X[:, 2]
    age_mean_glm, age_sd_glm = _age.mean(), _age.std()
    year_mean_glm, year_sd_glm = _year.mean(), _year.std()

    _ref_surg_idx = surgeon_levels.index("Dr. James Andrews")
    _n_surg = len(surgeon_levels)
    _non_ref = [i for i in range(_n_surg) if i != _ref_surg_idx]
    _surg_codes = surv_X[:, 6].astype(int)
    _surg_dummies = np.zeros((surv_X.shape[0], _n_surg - 1))
    for _col, _level_idx in enumerate(_non_ref):
        _surg_dummies[:, _col] = (_surg_codes == _level_idx).astype(float)

    surv_X_glm = np.column_stack(
        [
            (_age - age_mean_glm) / age_sd_glm,
            (_year - year_mean_glm) / year_sd_glm,
            surv_X[:, 3],
            surv_X[:, 4],
            surv_X[:, 5],
            _surg_dummies,
        ]
    )

    glm_non_ref_surg = _non_ref
    f"surv_X_glm shape {surv_X_glm.shape}, q_idx range [{q_idx.min()}, {q_idx.max()}]"
    return (
        age_mean_glm,
        age_sd_glm,
        glm_non_ref_surg,
        q_idx,
        surv_X_glm,
        year_mean_glm,
        year_sd_glm,
    )


@app.cell
def _(RANDOM_SEED, pm, q_idx, surv_X_glm, surv_y):
    with pm.Model() as model_glm:
        X_glm_data = pm.Data("X_glm_data", surv_X_glm)
        q_idx_data = pm.Data("q_idx_data", q_idx, dims="row")
        beta = pm.Normal("beta", 0.0, 1.0, shape=surv_X_glm.shape[1])
        alpha_q = pm.Normal("alpha_q", -3.0, 1.5, shape=12)
        eta_glm = alpha_q[q_idx_data] + pm.math.dot(X_glm_data, beta)
        h_glm = pm.Deterministic("h_glm", 1.0 - pm.math.exp(-pm.math.exp(eta_glm)))
        pm.Bernoulli("event_glm", p=h_glm, observed=surv_y, shape=h_glm.shape)
        idata_glm = pm.sample(random_seed=RANDOM_SEED)
    return idata_glm, model_glm


@app.cell
def _(az, idata_glm):
    _summary_glm = az.summary(
        idata_glm,
        var_names=["beta", "alpha_q"],
        kind="diagnostics",
        round_to=3,
    )
    (
        f"GLM diagnostics -- max R-hat: {_summary_glm['r_hat'].max():.3f}, "
        f"min ESS bulk: {int(_summary_glm['ess_bulk'].min())}, "
        f"min ESS tail: {int(_summary_glm['ess_tail'].min())}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuring the PGBART step

    `pm.sample` auto-registers `pmb.PGBART` as the step method for any
    BART random variable in the model, but the sampler is configurable
    when the defaults don't suit the problem. Two kwargs matter most:

    - **`num_particles`** (default `10`): number of particles used in
      the conditional sequential Monte Carlo proposal. More particles
      give better proposals at proportional per-step wall-clock cost.
      The fit below uses `num_particles=20`, trading roughly 2x
      per-sweep compute for sharper tree proposals on this 12k-row
      person-month expansion.
    - **`batch`** (default `(0.1, 0.1)`): a `(tune_fraction,
      post_tune_fraction)` pair giving the fraction of the `m` trees
      refit per Gibbs sweep. This is a mixing knob, not a speed knob:
      raising `batch` refits more trees per sweep (so each sweep costs
      *more*, not less) but lets the chain explore tree space in
      fewer total sweeps. The fit below uses `(0.1, 0.15)`,
      refitting slightly more trees once tuning ends to help mixing
      on the time axis.

    The fit cell below passes both knobs through an explicit
    `step=pmb.PGBART(vars=[eta], num_particles=20, batch=(0.1, 0.15))`.
    """)
    return


@app.cell
def _(RANDOM_SEED, pm, pmb, split_rules, surv_X, surv_y):
    with pm.Model() as model_surv:
        X_data = pm.Data("X_data", surv_X)
        eta = pmb.BART(
            "eta",
            X=X_data,
            Y=surv_y.astype(float),
            m=100,
            split_rules=split_rules,
        )
        h = pm.Deterministic("h", 1.0 - pm.math.exp(-pm.math.exp(eta)))
        pm.Bernoulli("event", p=h, observed=surv_y, shape=h.shape)
        idata_surv = pm.sample(
            random_seed=RANDOM_SEED,
            step=pmb.PGBART(vars=[eta], num_particles=20, batch=(0.1, 0.15)),
        )
    return eta, idata_surv, model_surv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same convergence view as before, adapted to the BART fit:
    `az.plot_convergence_dist` shows the distributions of R-hat and ESS
    across the elements of the latent `eta`.
    """)
    return


@app.cell
def _(az, idata_surv):
    az.plot_convergence_dist(idata_surv, var_names=["eta"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Covariate contrast profiles

    To see what the two fits imply, we evaluate both posteriors at a
    grid of covariate profiles. The reference profile holds each
    covariate at its training-data mode or median for the broadened
    (all-levels, all-positions) population: age 22, surgery year 2012,
    mid-season date, right-handed, first surgery, Dr. James Andrews.
    Each panel below sweeps one covariate while pinning the rest at the
    reference values; the age contrast uses {20, 30, 40} since the data
    spans ages 13--47.
    Every profile is encoded twice — `X_profiles` in the raw 7-column
    BART design and `X_profiles_glm` in the wide GLM design — and
    `q_idx_profiles` maps each profile row to its quarter for the
    GLM's $\alpha_q$ baseline. Following the `pm.Data` / `pm.set_data`
    prediction pattern from the regression notebook, we swap these profile
    rows into the fitted models before sampling. For the GLM, `h_glm` is a Deterministic of `alpha_q` and `beta`,
    so we also reattach `q_idx_data` and sample hazard draws directly.
    """)
    return


@app.cell
def _(
    age_mean_glm,
    age_sd_glm,
    glm_non_ref_surg,
    np,
    surgeon_levels,
    throws_levels,
    year_mean_glm,
    year_sd_glm,
):
    AGE_REF = 22
    YEAR_REF = 2012
    DOY_REF = 0.4
    THROWS_R = throws_levels.index("R")
    THROWS_L = throws_levels.index("L")
    REV_REF = 0
    SURG_REF = surgeon_levels.index("Dr. James Andrews")
    SURG_UNKNOWN = surgeon_levels.index("Unknown")
    times = np.arange(1, 37)

    def _profile_bart(age, year, doy, throws, rev, surg):
        n = len(times)
        return np.column_stack(
            [
                times,
                np.full(n, age),
                np.full(n, year),
                np.full(n, doy),
                np.full(n, throws),
                np.full(n, rev),
                np.full(n, surg),
            ]
        )

    def _profile_glm(age, year, doy, throws, rev, surg):
        n = len(times)
        surg_row = np.zeros(len(glm_non_ref_surg))
        if surg in glm_non_ref_surg:
            surg_row[glm_non_ref_surg.index(surg)] = 1.0
        return np.column_stack(
            [
                np.full(n, (age - age_mean_glm) / age_sd_glm),
                np.full(n, (year - year_mean_glm) / year_sd_glm),
                np.full(n, doy),
                np.full(n, throws),
                np.full(n, rev),
                np.tile(surg_row, (n, 1)),
            ]
        )

    _specs = {
        "age_20": (20, YEAR_REF, DOY_REF, THROWS_R, REV_REF, SURG_REF),
        "age_30": (30, YEAR_REF, DOY_REF, THROWS_R, REV_REF, SURG_REF),
        "age_40": (40, YEAR_REF, DOY_REF, THROWS_R, REV_REF, SURG_REF),
        "year_1995": (AGE_REF, 1995, DOY_REF, THROWS_R, REV_REF, SURG_REF),
        "year_2020": (AGE_REF, 2020, DOY_REF, THROWS_R, REV_REF, SURG_REF),
        "throws_R": (AGE_REF, YEAR_REF, DOY_REF, THROWS_R, REV_REF, SURG_REF),
        "throws_L": (AGE_REF, YEAR_REF, DOY_REF, THROWS_L, REV_REF, SURG_REF),
        "surg_andrews": (AGE_REF, YEAR_REF, DOY_REF, THROWS_R, REV_REF, SURG_REF),
        "surg_unknown": (AGE_REF, YEAR_REF, DOY_REF, THROWS_R, REV_REF, SURG_UNKNOWN),
        "rev_0": (AGE_REF, YEAR_REF, DOY_REF, THROWS_R, 0, SURG_REF),
        "rev_1": (AGE_REF, YEAR_REF, DOY_REF, THROWS_R, 1, SURG_REF),
    }
    profiles = {k: _profile_bart(*args) for k, args in _specs.items()}
    profiles_glm = {k: _profile_glm(*args) for k, args in _specs.items()}

    profile_names = list(profiles.keys())
    X_profiles = np.concatenate([profiles[k] for k in profile_names], axis=0)
    X_profiles_glm = np.concatenate([profiles_glm[k] for k in profile_names], axis=0)
    q_idx_profiles = np.tile(((times - 1) // 3).astype(int), len(profile_names))
    return X_profiles, X_profiles_glm, profile_names, q_idx_profiles, times


@app.cell
def _(RANDOM_SEED, X_profiles, idata_surv, model_surv, pm):
    with model_surv:
        pm.set_data({"X_data": X_profiles})
        pp_surv = pm.sample_posterior_predictive(
            idata_surv,
            var_names=["h"],
            sample_vars=["eta"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_surv,)


@app.cell
def _(RANDOM_SEED, X_profiles_glm, idata_glm, model_glm, pm, q_idx_profiles):
    with model_glm:
        pm.set_data({"X_glm_data": X_profiles_glm, "q_idx_data": q_idx_profiles})
        pp_glm = pm.sample_posterior_predictive(
            idata_glm,
            var_names=["h_glm"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_glm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Survival curves

    From hazard draws to survival curves: each profile contributes 36
    consecutive rows of the posterior predictive, which are sliced out
    and accumulated via $S(t) = \prod_{s \le t} (1 - h(s))$ — once for
    the BART draws and once for the GLM draws. We stack chain and draw into a
    single sample axis before building the survival curves. As a no-covariate
    reference, a marginal Kaplan-Meier curve computed from the
    subject-level data (empirical per-month hazard, no model) is
    overlaid as a dotted gray line on every panel.

    Interpret these curves under the right-censoring assumption above:
    career-ending non-returns treated as censored can bias survival upward.
    """)
    return


@app.cell
def _(np, pp_glm, pp_surv, profile_names, times):
    _h_draws = pp_surv.predictions["h"].stack(sample=("chain", "draw")).values
    _h_glm_draws = pp_glm.predictions["h_glm"].stack(sample=("chain", "draw")).values
    _n_t = len(times)
    S_curves = {}
    h_curves = {}
    S_glm_curves = {}
    h_glm_curves = {}
    for _i, _name in enumerate(profile_names):
        _slc_bart = _h_draws[_i * _n_t : (_i + 1) * _n_t, :].T
        _slc_glm = _h_glm_draws[_i * _n_t : (_i + 1) * _n_t, :].T
        h_curves[_name] = _slc_bart
        h_glm_curves[_name] = _slc_glm
        S_curves[_name] = np.cumprod(1 - _slc_bart, axis=1)
        S_glm_curves[_name] = np.cumprod(1 - _slc_glm, axis=1)
    return S_curves, S_glm_curves, h_curves, h_glm_curves


@app.cell
def _(np, tj_df):
    def km_survival(time_months, event, max_t=36):
        t_bin = np.maximum(1, np.ceil(time_months)).astype(int)
        h = np.zeros(max_t)
        for t in range(1, max_t + 1):
            at_risk = (t_bin >= t).sum()
            events_t = ((t_bin == t) & (event == 1)).sum()
            h[t - 1] = events_t / max(at_risk, 1)
        return np.cumprod(1 - h)

    S_km = km_survival(
        tj_df["time_months"].to_numpy(),
        tj_df["event"].to_numpy(),
        max_t=36,
    )
    return S_km, km_survival


@app.cell(hide_code=True)
def _(S_curves, S_glm_curves, S_km, np, plt, times):
    def _band(curves):
        return curves.mean(axis=0), np.quantile(curves, [0.05, 0.95], axis=0)

    def _draw_bart(ax, key, label, color):
        _m, (_lo, _hi) = _band(S_curves[key])
        ax.step(times, _m, where="post", color=color, label=f"BART {label}")
        ax.fill_between(times, _lo, _hi, step="post", color=color, alpha=0.2)

    def _draw_glm(ax, key, color):
        _m = S_glm_curves[key].mean(axis=0)
        ax.step(times, _m, where="post", color=color, linestyle="--", linewidth=1.4)

    _fig, _axes = plt.subplot_mosaic(
        [["age", "era", "throws"], ["surg", "rev", "."]],
        figsize=(13, 8),
        sharey=True,
    )

    _ax = _axes["age"]
    for _age, _color in zip([20, 30, 40], ["#4c72b0", "#8172b2", "#c44e52"]):
        _key = f"age_{_age}"
        _draw_bart(_ax, _key, f"age {_age}", _color)
        _draw_glm(_ax, _key, _color)
    _ax.set_title("Age (year=2012, Andrews, R, first TJ)")

    _ax = _axes["era"]
    for _yr, _color in zip([1995, 2020], ["#4c72b0", "#c44e52"]):
        _key = f"year_{_yr}"
        _draw_bart(_ax, _key, f"year {_yr}", _color)
        _draw_glm(_ax, _key, _color)
    _ax.set_title("Surgery year (age=22, Andrews, R, first TJ)")

    _ax = _axes["throws"]
    for _key, _label, _color in zip(
        ["throws_R", "throws_L"], ["throws R", "throws L"], ["#4c72b0", "#c44e52"]
    ):
        _draw_bart(_ax, _key, _label, _color)
        _draw_glm(_ax, _key, _color)
    _ax.set_title("Throwing hand (age=22, year=2012, Andrews, first TJ)")

    _ax = _axes["surg"]
    for _key, _label, _color in zip(
        ["surg_andrews", "surg_unknown"],
        ["Andrews", "Unknown"],
        ["#4c72b0", "#c44e52"],
    ):
        _draw_bart(_ax, _key, _label, _color)
        _draw_glm(_ax, _key, _color)
    _ax.set_title("Surgeon (age=22, year=2012, R, first TJ)")

    _ax = _axes["rev"]
    for _key, _label, _color in zip(
        ["rev_0", "rev_1"], ["first TJ", "revision"], ["#4c72b0", "#c44e52"]
    ):
        _draw_bart(_ax, _key, _label, _color)
        _draw_glm(_ax, _key, _color)
    _ax.set_title("Revision (age=22, year=2012, Andrews, R)")

    for _key in ["age", "era", "throws", "surg", "rev"]:
        _ax = _axes[_key]
        _ax.step(
            times,
            S_km,
            where="post",
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="KM (marginal)",
        )
        _ax.set_xlabel("months since surgery")
        _ax.set_ylim(0, 1.02)
        _ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=1)

    _axes["age"].set_ylabel(r"$S(t \mid x)$")
    _axes["surg"].set_ylabel(r"$S(t \mid x)$")
    _fig.suptitle(
        "Predicted survival -- BART (solid, 90% band) vs cloglog GLM (dashed) vs marginal KM (dotted)"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hazards behind the survival curves

    Survival is the cumulative product
    $S(t \mid x) = \prod_{s \le t}(1 - h(s \mid x))$, but what BART
    actually models is the per-month hazard $h(t \mid x)$ on each
    person-month row. Plotting the hazard directly shows the shape that
    the cumulative product smooths out: it should rise to a peak around
    months 12 to 18 (the typical Tommy John recovery window) and decay
    afterward.

    The GLM dashed line is a step function over quarters by construction
    (the only flexibility we gave the baseline is `alpha_q` for the 12
    three-month bins). BART can split on $t$ wherever it likes, so its
    curve is allowed to break the quarter grid.
    """)
    return


@app.cell(hide_code=True)
def _(h_curves, h_glm_curves, np, plt, times):
    def _band_h(curves):
        return curves.mean(axis=0), np.quantile(curves, [0.05, 0.95], axis=0)

    def _draw_haz_bart(ax, key, label, color):
        _m, (_lo, _hi) = _band_h(h_curves[key])
        ax.step(times, _m, where="post", color=color, label=f"BART {label}")
        ax.fill_between(times, _lo, _hi, step="post", color=color, alpha=0.2)

    def _draw_haz_glm(ax, key, color):
        _m = h_glm_curves[key].mean(axis=0)
        ax.step(times, _m, where="post", color=color, linestyle="--", linewidth=1.4)

    _fig_h, _axes_h = plt.subplot_mosaic(
        [["age", "era", "throws"], ["surg", "rev", "."]],
        figsize=(13, 8),
        sharey=True,
    )

    _ax = _axes_h["age"]
    for _age, _color in zip([20, 30, 40], ["#4c72b0", "#8172b2", "#c44e52"]):
        _key = f"age_{_age}"
        _draw_haz_bart(_ax, _key, f"age {_age}", _color)
        _draw_haz_glm(_ax, _key, _color)
    _ax.set_title("Age")

    _ax = _axes_h["era"]
    for _yr, _color in zip([1995, 2020], ["#4c72b0", "#c44e52"]):
        _key = f"year_{_yr}"
        _draw_haz_bart(_ax, _key, f"year {_yr}", _color)
        _draw_haz_glm(_ax, _key, _color)
    _ax.set_title("Surgery year")

    _ax = _axes_h["throws"]
    for _key, _label, _color in zip(
        ["throws_R", "throws_L"], ["throws R", "throws L"], ["#4c72b0", "#c44e52"]
    ):
        _draw_haz_bart(_ax, _key, _label, _color)
        _draw_haz_glm(_ax, _key, _color)
    _ax.set_title("Throwing hand")

    _ax = _axes_h["surg"]
    for _key, _label, _color in zip(
        ["surg_andrews", "surg_unknown"],
        ["Andrews", "Unknown"],
        ["#4c72b0", "#c44e52"],
    ):
        _draw_haz_bart(_ax, _key, _label, _color)
        _draw_haz_glm(_ax, _key, _color)
    _ax.set_title("Surgeon")

    _ax = _axes_h["rev"]
    for _key, _label, _color in zip(
        ["rev_0", "rev_1"], ["first TJ", "revision"], ["#4c72b0", "#c44e52"]
    ):
        _draw_haz_bart(_ax, _key, _label, _color)
        _draw_haz_glm(_ax, _key, _color)
    _ax.set_title("Revision")

    for _key in ["age", "era", "throws", "surg", "rev"]:
        _ax = _axes_h[_key]
        _ax.set_xlabel("months since surgery")
        _ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1)

    _axes_h["age"].set_ylabel(r"$h(t \mid x)$")
    _axes_h["surg"].set_ylabel(r"$h(t \mid x)$")
    _fig_h.suptitle(
        "Per-month hazard -- BART (solid, 90% band) vs cloglog GLM (dashed)"
    )
    _fig_h.tight_layout()
    _fig_h
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the proportional-hazards assumption hold?

    The GLM comparator above assumes a constant interval cumulative hazard
    ratio over time. BART does not have to: the same cloglog link is wrapped
    around a tree-sum function of both time and covariates.

    The plot below contrasts the two on a single comparison: 40-year-old
    vs 20-year-old pitcher, all other covariates pinned at the
    reference. The GLM line is flat (its interval HR is `exp(beta_age * 20 /
    sd_age)`); BART's ribbon is allowed to be a curve. If the ribbon
    stays inside the GLM band, proportional hazards approximately holds
    for this contrast. If it drifts up or down as $t$ grows, BART has
    found a non-PH effect that Cox PH would miss.
    """)
    return


@app.cell(hide_code=True)
def _(age_sd_glm, h_curves, idata_glm, np, plt, times):
    def _interval_hazard_ratio(_h_num, _h_den):
        _lambda_num = -np.log1p(-np.clip(_h_num, 0.0, 1.0 - 1e-12))
        _lambda_den = -np.log1p(-np.clip(_h_den, 0.0, 1.0 - 1e-12))
        return _lambda_num / np.clip(_lambda_den, 1e-12, None)

    _bart_hr_draws = _interval_hazard_ratio(h_curves["age_40"], h_curves["age_20"])
    _bart_hr_med = np.median(_bart_hr_draws, axis=0)
    _bart_hr_lo, _bart_hr_hi = np.quantile(_bart_hr_draws, [0.05, 0.95], axis=0)

    _beta_age = idata_glm.posterior["beta"].sel(beta_dim_0=0).values.ravel()
    _glm_hr_draws = np.exp(_beta_age * (40 - 20) / age_sd_glm)
    _glm_hr_med = float(np.median(_glm_hr_draws))
    _glm_hr_lo, _glm_hr_hi = np.quantile(_glm_hr_draws, [0.05, 0.95])

    _fig_hr, _ax_hr = plt.subplots(figsize=(8, 4.8))
    _ax_hr.fill_between(
        times,
        _bart_hr_lo,
        _bart_hr_hi,
        step="post",
        color="#4c72b0",
        alpha=0.2,
        label="BART 90% CrI",
    )
    _ax_hr.step(times, _bart_hr_med, where="post", color="#4c72b0", label="BART median")
    _ax_hr.axhline(
        _glm_hr_med,
        color="#c44e52",
        linestyle="--",
        linewidth=1.4,
        label="cloglog GLM HR",
    )
    _ax_hr.axhspan(_glm_hr_lo, _glm_hr_hi, color="#c44e52", alpha=0.12)
    _ax_hr.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    _ax_hr.set_xlabel("months since surgery")
    _ax_hr.set_ylabel(
        r"HR$(t)$ = $\Lambda(t|\text{age}=40) / \Lambda(t|\text{age}=20)$"
    )
    _ax_hr.set_title(
        f"Time-varying interval hazard ratio: 40 vs 20\ncloglog GLM HR = {_glm_hr_med:.2f} (90% CrI [{_glm_hr_lo:.2f}, {_glm_hr_hi:.2f}])"
    )
    _ax_hr.legend(frameon=False, loc="upper left")
    _fig_hr.tight_layout()
    _fig_hr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior-predictive Kaplan-Meier check

    Posterior-predictive checks for survival data look different from
    their regression cousins. Right-censoring makes pointwise residuals
    awkward; the natural quantity to check is the marginal Kaplan-Meier
    curve. We sample 100 full cohorts from the posterior predictive (a
    simulated `event_{i,t}` at every person-month), reconstruct each
    subject's simulated time-to-event as the first month with a
    predicted return (censored at last observation otherwise), and
    compute the empirical KM curve on each simulated cohort.

    Before drawing those events we swap `X_data` back to the training
    person-month rows and keep `sample_vars=["eta"]` so `pymc-bart` walks the
    fitted trees rather than the prior mean. The subject-level reconstruction
    uses the same `max(1, ceil(time_months))` expansion rule as the data-prep
    cell, so the reconstructed row offsets line up with the original
    person-month table.

    If the observed KM lands inside the cloud of simulated KMs, BART is
    calibrated against the marginal survival distribution implied by the
    right-censoring model. A systematic deviation indicates structural
    mis-calibration that the contrast panels would not surface; career-ending
    competing events remain outside this check.
    """)
    return


@app.cell
def _(RANDOM_SEED, idata_surv, model_surv, pm, surv_X):
    with model_surv:
        pm.set_data({"X_data": surv_X})
        pp_train = pm.sample_posterior_predictive(
            idata_surv,
            var_names=["event"],
            sample_vars=["eta"],
            random_seed=RANDOM_SEED,
        )
    return (pp_train,)


@app.cell(hide_code=True)
def _(RANDOM_SEED, S_km, km_survival, np, plt, pp_train, times, tj_df):
    _n_periods = np.maximum(1, np.ceil(tj_df["time_months"].to_numpy())).astype(int)
    _subject_starts = np.concatenate([[0], np.cumsum(_n_periods)])
    _n_subjects = len(_n_periods)
    _event_draws = (
        pp_train.posterior_predictive["event"].stack(sample=("chain", "draw")).values
    )
    _rng_ppc = np.random.default_rng(RANDOM_SEED)
    _n_sim = 100
    _draw_idx = _rng_ppc.choice(_event_draws.shape[1], size=_n_sim, replace=False)

    def _sim_km(events_flat):
        sim_time = np.empty(_n_subjects, dtype=int)
        sim_event = np.empty(_n_subjects, dtype=int)
        for _i in range(_n_subjects):
            _s, _e = _subject_starts[_i], _subject_starts[_i + 1]
            _segment = events_flat[_s:_e]
            _first = np.argmax(_segment)
            if _segment[_first] == 1:
                sim_time[_i] = _first + 1
                sim_event[_i] = 1
            else:
                sim_time[_i] = len(_segment)
                sim_event[_i] = 0
        return km_survival(sim_time.astype(float), sim_event, max_t=36)

    _sim_kms = np.array([_sim_km(_event_draws[:, _d]) for _d in _draw_idx])

    _fig_pk, _ax_pk = plt.subplots(figsize=(8, 4.8))
    for _km in _sim_kms:
        _ax_pk.step(
            times, _km, where="post", color="#4c72b0", alpha=0.08, linewidth=0.7
        )
    _ax_pk.step(
        times,
        S_km,
        where="post",
        color="black",
        linewidth=2.0,
        label="observed KM",
    )
    _ax_pk.step(
        times,
        _sim_kms.mean(axis=0),
        where="post",
        color="#4c72b0",
        linewidth=1.6,
        linestyle="--",
        label="BART posterior-predictive mean KM",
    )
    _ax_pk.set_xlabel("months since surgery")
    _ax_pk.set_ylabel("S(t)")
    _ax_pk.set_ylim(0, 1.02)
    _ax_pk.legend(frameon=False, loc="lower left")
    _ax_pk.set_title(f"Posterior-predictive KM check ({_n_sim} simulated cohorts)")
    _fig_pk.tight_layout()
    _fig_pk
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Restricted mean survival time

    The contrast panels are easy to read but hard to quote. Restricted
    mean survival time at 36 months,
    $\text{RMST}(36) = \sum_{t=1}^{36} S(t \mid x),$
    collapses each curve into one number: the expected number of months
    out of the next three years the pitcher spends *not* yet returned.
    Pre-summing over `t` lets us display BART and the cloglog GLM side by
    side with 90% credible intervals, so the disagreements between the
    two models become numerical rather than purely visual. Interpret the
    values under the right-censoring assumption; competing career-ending
    non-returns can push them upward.
    """)
    return


@app.cell
def _(S_curves, S_glm_curves, np, pl, profile_names):
    def _rmst_summary(curves):
        _vals = curves.sum(axis=1)
        return np.mean(_vals), np.quantile(_vals, 0.05), np.quantile(_vals, 0.95)

    _rows = []
    for _name in profile_names:
        _b_mean, _b_lo, _b_hi = _rmst_summary(S_curves[_name])
        _g_mean, _g_lo, _g_hi = _rmst_summary(S_glm_curves[_name])
        _rows.append(
            {
                "profile": _name,
                "bart_rmst": round(_b_mean, 2),
                "bart_lo": round(_b_lo, 2),
                "bart_hi": round(_b_hi, 2),
                "glm_rmst": round(_g_mean, 2),
                "glm_lo": round(_g_lo, 2),
                "glm_hi": round(_g_hi, 2),
                "bart_minus_glm": round(_b_mean - _g_mean, 2),
            }
        )
    rmst_table = pl.DataFrame(_rows)
    rmst_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which covariates drive the hazard?

    **Forward** inclusion ("VI", the default) counts how often each
    variable is used as a splitting variable across the posterior trees.
    The classification notebook contrasts this with backward elimination,
    which refits restricted submodels to disentangle correlated features;
    here the fast forward ranking is enough.

    The seven columns are time `t`, `age`, `surgery_year`,
    `surgery_doy_frac`, `throws`, `revision`, and `surgeon_group`. We
    expect `t` to dominate (the hazard is highly time-shaped).
    """)
    return


@app.cell
def _(RANDOM_SEED, eta, feature_names, idata_surv, model_surv, plt, pm, pmb, surv_X):
    with model_surv:
        pm.set_data({"X_data": surv_X})
        _vi_fwd = pmb.compute_variable_importance(
            idata_surv, eta, surv_X, method="VI", random_seed=RANDOM_SEED
        )

    _fig_vi, _ax_vi = plt.subplots(figsize=(7, 4.5))
    pmb.plot_variable_importance(_vi_fwd, labels=feature_names, ax=_ax_vi)
    _ax_vi.set_title("Forward (inclusion-frequency)")
    _fig_vi.tight_layout()
    _fig_vi
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Partial dependence

    Partial dependence plots show the marginal effect of each feature
    on the latent log interval hazard, averaging the tree-sum
    prediction over the observed values of the remaining features.
    `var_discrete=[0, 4, 5, 6]` marks the integer-coded `t`, `throws`,
    `revision`, and `surgeon` columns so they render as ticks rather
    than smooth curves; `age`, `surgery_year`, and `surgery_doy_frac`
    stay continuous.
    """)
    return


@app.cell
def _(eta, model_surv, pm, pmb, surv_X, surv_y):
    with model_surv:
        pm.set_data({"X_data": surv_X})
        pmb.plot_pdp(
            bartrv=eta,
            X=surv_X,
            Y=surv_y.astype(float),
            var_discrete=[0, 4, 5, 6],
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From PDP to ICE: individual conditional expectations

    The partial dependence plots above marginalize each covariate's
    effect *across* the person-month rows: one curve per variable,
    averaged over all individuals. That is the right view for a
    global "does `age` matter?" question, but it hides
    heterogeneity, the case where individualised risk matters and
    the framing survival analysis exists to address.

    **Individual conditional expectation (ICE)** plots address
    this: one curve per *instance*, so where the effect of a
    covariate varies across the population, you see a fan of
    curves instead of a single average. For Tommy John recovery
    this matters at the per-pitcher level: a 22-year-old reliever
    operated on by ElAttrache in 2020 looks nothing like a
    34-year-old starter operated on by an unknown surgeon in 1998,
    and we want to see those trajectories separately.

    `centered=True` below subtracts each curve's value at the
    first `t`, so all curves start at zero and the *spread of
    slopes* (the heterogeneity we care about) is the visible
    signal rather than the per-instance baseline level.
    """)
    return


@app.cell
def _(
    RANDOM_SEED,
    eta,
    model_surv,
    pm,
    pmb,
    surv_X,
    surv_y,
):
    with model_surv:
        pm.set_data({"X_data": surv_X})
        pmb.plot_ice(
            bartrv=eta,
            X=surv_X,
            Y=surv_y.astype(float),
            var_discrete=[0],
            instances=30,
            centered=True,
            random_seed=RANDOM_SEED,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Survival wrap-up

    Discrete-time survival turns time-to-event modeling into person-month
    binary modeling: each row asks whether the player returned this month,
    conditional on still being at risk. BART and the cloglog GLM share the
    same hazard link, but differ in the function class inside the link:
    linear additive effects for the GLM, a flexible tree sum for BART.

    The plots answer different stakeholder questions. Survival curves show
    the probability of still not returned by month $t$; hazards show the
    month-by-month return window; interval hazard ratios test proportional
    hazards; RMST compresses a curve into one quoteable number. BART can reveal
    non-PH patterns that a Cox-style model would flatten, while ICE curves
    connect the model back to individualized risk. The competing-risks caveat
    remains: treating career-ending non-returns as censoring can bias survival
    curves upward, so a subdistribution-hazard model is the next step for a
    production analysis.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
