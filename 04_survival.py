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

    We model **discrete-time hazards**: at each integer month $t$
    post-surgery the return event occurs with probability
    $$h(t \mid x) = 1 - \exp\!\bigl(-\exp(g(t, x))\bigr),$$
    where $g$ is a sum of trees and the link is complementary log-log
    (cloglog). The cloglog link is the discrete-time analogue of the
    Cox proportional-hazards model: if $g$ were linear in $x$, the
    hazard ratio between any two profiles would be constant in $t$.
    BART makes $g$ a sum of trees instead, so the same likelihood can
    represent time-varying hazard ratios. Each subject contributes
    one row per month they remain at risk; the outcome is "did they
    return this month?". That is exactly the binary classification
    setup from notebook 3, applied to expanded person-time data.
    Survival follows from the hazards:
    $$S(t \mid x) = \prod_{s \le t} \bigl(1 - h(s \mid x)\bigr).$$

    The notebook runs in two passes:

    1. **A synthetic warmup** with a known data-generating process.
       The DGP is constructed to bake in a nonlinear age effect (knee
       at age 30) and a non-proportional-hazards treatment effect
       (HR(t) grows from 1 to 2 across the 36-month horizon) — both
       structures a no-interaction linear cloglog GLM has no
       parameters to fit. We fit BART and a cloglog GLM side by
       side and verify the BART machinery recovers what we put in.
    2. **The real Tommy John data**, using the same machinery. Once
       we trust BART against a known truth, we use it as a flexible
       alternative to the standard Cox-PH-style analysis on real
       data whose true structure is unknown.
    """)
    return


@app.cell
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
    rng = np.random.default_rng(RANDOM_SEED)
    return RANDOM_SEED, az, np, pl, plt, pm, pmb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Warm-up: a constructed machinery check

    Before fitting BART on real Tommy John data, we check the
    machinery end-to-end on a synthetic dataset where the truth is
    known. This is not a fair contest between BART and the cloglog
    GLM. The DGP below mirrors the real-data feature layout (person-
    time discrete hazards on a 36-month horizon, 7 covariates) and
    *deliberately* bakes in two effects that a no-interaction linear
    cloglog GLM cannot represent at all:

    1. **Nonlinear age effect with a knee at age 30.** No effect below
       30; log-hazard increases by 0.05 per year above. A linear
       predictor in `age` will smear this across the full range.
    2. **Non-proportional-hazards treatment effect.** HR(t) starts at
       1.0 at month 1 and grows to 2.0 by month 36, an interaction
       between the treatment indicator and time. A GLM without a
       time-by-treatment interaction term has no parameter to
       represent it; the cloglog link forces a constant HR.

    Three other covariates carry weaker structure (era shift, surgeon
    clustering, a fixed-effect multiplier on a 5-level surgeon
    category). Two covariates (day-of-year, throws) are pure noise.

    Because the DGP is constructed to violate the GLM's linear-additive
    assumption, the GLM is *guaranteed* to miss the time-varying HR
    and the age knee. The question we're answering is therefore not
    "which model wins" but "does the BART machinery recover what we
    put in?" — specifically:

    - `t`, treatment, age, and surgeon high in variable importance.
    - throws and doy at the bottom.
    - The growing HR(t) curve for treatment, traced cleanly against
      the GLM's flat constant.

    Once we trust the machinery against the known truth, we apply it
    to the real Tommy John data whose structure we do not know.
    """)
    return


@app.cell
def _(RANDOM_SEED, np, pmb):
    # Synthetic DGP mirroring the real Tommy John feature layout: t,
    # age, year, doy, throws, rev, surgeon. The hazard is built from a
    # baseline shape h_0(t) (lognormal-style bell, peak ~0.08 around
    # t=14) multiplied by per-covariate effects. Two of those effects
    # violate the cloglog GLM's linear-additive assumption: the age
    # knee at 30, and the treatment-by-time interaction.
    def _baseline_hazard(t):
        return 0.001 + 0.08 * np.exp(-0.5 * ((np.log(t) - np.log(14)) ** 2) / 0.4)

    def simulate_survival(n_subjects=500, max_t=36, seed=0):
        sim_rng = np.random.default_rng(seed)
        age = sim_rng.uniform(20, 40, n_subjects)
        year = sim_rng.uniform(1995, 2025, n_subjects)
        doy = sim_rng.uniform(0, 1, n_subjects)
        throws = sim_rng.integers(0, 2, n_subjects)
        rev = (sim_rng.random(n_subjects) < 0.3).astype(int)
        surgeon = sim_rng.integers(0, 5, n_subjects)
        surgeon_mult = np.array([1.0, 0.7, 1.3, 0.9, 1.1])

        rows = []
        for i in range(n_subjects):
            age_eff = np.exp(max(0.0, age[i] - 30) * 0.05)
            year_eff = np.exp(0.015 * (year[i] - 2010))
            surg_eff = surgeon_mult[surgeon[i]]
            for t in range(1, max_t + 1):
                base = _baseline_hazard(t)
                treat_eff = 1.0 + rev[i] * t / max_t
                h = min(base * age_eff * year_eff * surg_eff * treat_eff, 0.99)
                event = int(sim_rng.random() < h)
                rows.append(
                    (t, age[i], year[i], doy[i], throws[i], rev[i], surgeon[i], event)
                )
                if event == 1:
                    break
        arr = np.array(rows, dtype=float)
        return arr[:, :7], arr[:, 7].astype(int)

    sim_X, sim_y = simulate_survival(n_subjects=500, max_t=36, seed=RANDOM_SEED)
    sim_q_idx = ((sim_X[:, 0].astype(int) - 1) // 3).astype(int)
    sim_n_subjects = 500

    # GLM design matrix for the synthetic data: standardize age and year,
    # binarize throws / rev, dummy-code the 5-level surgeon (reference =
    # level 0). doy passes through as the [0, 1] noise feature.
    sim_age_mean, sim_age_sd = sim_X[:, 1].mean(), sim_X[:, 1].std()
    sim_year_mean, sim_year_sd = sim_X[:, 2].mean(), sim_X[:, 2].std()

    sim_surg_codes = sim_X[:, 6].astype(int)
    sim_surg_dummies = np.zeros((sim_X.shape[0], 4))
    for _col, _level in enumerate([1, 2, 3, 4]):
        sim_surg_dummies[:, _col] = (sim_surg_codes == _level).astype(float)

    sim_X_glm = np.column_stack(
        [
            (sim_X[:, 1] - sim_age_mean) / sim_age_sd,
            (sim_X[:, 2] - sim_year_mean) / sim_year_sd,
            sim_X[:, 3],
            sim_X[:, 4],
            sim_X[:, 5],
            sim_surg_dummies,
        ]
    )
    sim_glm_feature_names = [
        "age_std",
        "year_std",
        "doy",
        "throws",
        "rev",
        "surg_1",
        "surg_2",
        "surg_3",
        "surg_4",
    ]
    sim_split_rules = (
        [pmb.ContinuousSplitRule] * 4
        + [pmb.OneHotSplitRule] * 2
        + [pmb.SubsetSplitRule]
    )
    sim_feature_names = ["t", "age", "year", "doy", "throws", "rev", "surgeon"]
    f"sim_X shape {sim_X.shape}, events {int(sim_y.sum())}, sim_X_glm cols {sim_X_glm.shape[1]}"
    return (
        sim_X,
        sim_X_glm,
        sim_age_mean,
        sim_age_sd,
        sim_q_idx,
        sim_split_rules,
        sim_y,
        sim_year_mean,
        sim_year_sd,
    )


@app.cell
def _(RANDOM_SEED, pm, pmb, sim_X, sim_split_rules, sim_y):
    # Synthetic BART fit. m=50 (instead of the 100 we use on real
    # data) gives a faster fit appropriate for a warmup demo; the
    # synthetic dataset is small enough that 50 trees suffice. The
    # PGBART step config (num_particles, batch) matches the real fit.
    with pm.Model() as model_sim_bart:
        X_sim_data = pm.Data("X_sim_data", sim_X)
        eta_sim = pmb.BART(
            "eta_sim",
            X=X_sim_data,
            Y=sim_y.astype(float),
            m=50,
            split_rules=sim_split_rules,
        )
        h_sim = pm.Deterministic("h_sim", 1.0 - pm.math.exp(-pm.math.exp(eta_sim)))
        pm.Bernoulli("event_sim", p=h_sim, observed=sim_y, shape=h_sim.shape)
        idata_sim_bart = pm.sample(
            random_seed=RANDOM_SEED,
            step=pmb.PGBART(vars=[eta_sim], num_particles=20, batch=(0.1, 0.15)),
        )
    return idata_sim_bart, model_sim_bart


@app.cell
def _(RANDOM_SEED, pm, sim_X_glm, sim_q_idx, sim_y):
    # Synthetic GLM (cloglog discrete-time PH) on the same person-time
    # expansion. No treatment-by-time interaction term, so the model is
    # structurally incapable of representing the non-PH treatment effect
    # baked into the DGP. That structural failure is what BART corrects.
    with pm.Model() as model_sim_glm:
        X_sim_glm_data = pm.Data("X_sim_glm_data", sim_X_glm)
        sim_q_idx_data = pm.Data("sim_q_idx_data", sim_q_idx, dims="row")
        beta_sim = pm.Normal("beta_sim", 0.0, 1.0, shape=sim_X_glm.shape[1])
        alpha_q_sim = pm.Normal("alpha_q_sim", -3.0, 1.5, shape=12)
        eta_sim_glm = alpha_q_sim[sim_q_idx_data] + pm.math.dot(
            X_sim_glm_data, beta_sim
        )
        h_sim_glm = pm.Deterministic(
            "h_sim_glm", 1.0 - pm.math.exp(-pm.math.exp(eta_sim_glm))
        )
        pm.Bernoulli(
            "event_sim_glm", p=h_sim_glm, observed=sim_y, shape=h_sim_glm.shape
        )
        idata_sim_glm = pm.sample(random_seed=RANDOM_SEED)
    return idata_sim_glm, model_sim_glm


@app.cell
def _(np, sim_age_mean, sim_age_sd, sim_year_mean, sim_year_sd):
    # Treatment contrast profiles: 36-row sweeps at rev=0 vs rev=1,
    # all other covariates pinned at synthetic-data medians. The same
    # pair of profiles is encoded for BART and GLM so we can read off
    # HR(t) from each posterior predictive.
    sim_times = np.arange(1, 37)
    sim_ref = dict(age=30.0, year=2010.0, doy=0.5, throws=0, surgeon=0)

    def _sim_profile_bart(rev):
        n = len(sim_times)
        return np.column_stack(
            [
                sim_times,
                np.full(n, sim_ref["age"]),
                np.full(n, sim_ref["year"]),
                np.full(n, sim_ref["doy"]),
                np.full(n, sim_ref["throws"]),
                np.full(n, rev),
                np.full(n, sim_ref["surgeon"]),
            ]
        )

    def _sim_profile_glm(rev):
        n = len(sim_times)
        return np.column_stack(
            [
                np.full(n, (sim_ref["age"] - sim_age_mean) / sim_age_sd),
                np.full(n, (sim_ref["year"] - sim_year_mean) / sim_year_sd),
                np.full(n, sim_ref["doy"]),
                np.full(n, sim_ref["throws"]),
                np.full(n, rev),
                np.zeros((n, 4)),  # all surgeon dummies 0 = reference surgeon 0
            ]
        )

    sim_X_profiles = np.concatenate(
        [_sim_profile_bart(0), _sim_profile_bart(1)], axis=0
    )
    sim_X_profiles_glm = np.concatenate(
        [_sim_profile_glm(0), _sim_profile_glm(1)], axis=0
    )
    sim_q_idx_profiles = np.tile(((sim_times - 1) // 3).astype(int), 2)
    return sim_X_profiles, sim_X_profiles_glm, sim_q_idx_profiles, sim_times


@app.cell
def _(RANDOM_SEED, idata_sim_bart, model_sim_bart, pm, sim_X_profiles):
    with model_sim_bart:
        pm.set_data({"X_sim_data": sim_X_profiles})
        pp_sim_bart = pm.sample_posterior_predictive(
            idata_sim_bart,
            var_names=["h_sim"],
            sample_vars=["eta_sim"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_sim_bart,)


@app.cell
def _(
    RANDOM_SEED,
    idata_sim_glm,
    model_sim_glm,
    pm,
    sim_X_profiles_glm,
    sim_q_idx_profiles,
):
    with model_sim_glm:
        pm.set_data(
            {"X_sim_glm_data": sim_X_profiles_glm, "sim_q_idx_data": sim_q_idx_profiles}
        )
        pp_sim_glm = pm.sample_posterior_predictive(
            idata_sim_glm,
            var_names=["h_sim_glm"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_sim_glm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Did we recover the truth?

    The figure below compares three things at the treatment contrast
    (rev=1 vs rev=0):

    - **True HR(t)** (black dashed) = `1 + t/36` by construction.
    - **BART** (blue, with 90% CrI) reads HR(t) from its posterior
      predictive at the two profiles.
    - **Cloglog GLM** (red) reads `exp(beta_rev)` from the linear
      coefficient, a single number constant in t.

    A well-functioning BART fit traces the true curve closely. A
    well-functioning GLM lands somewhere near the *average* of the true
    HR over t (about 1.5), which is what a no-interaction PH model
    should do.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, pp_sim_bart, pp_sim_glm, sim_times):
    def _sim_band(curves):
        return curves.mean(axis=0), np.quantile(curves, [0.05, 0.95], axis=0)

    _h_sim_bart = (
        pp_sim_bart.predictions["h_sim"].stack(sample=("chain", "draw")).values
    )
    _h_sim_glm = (
        pp_sim_glm.predictions["h_sim_glm"].stack(sample=("chain", "draw")).values
    )
    _n_t = len(sim_times)
    _bart_rev0 = _h_sim_bart[:_n_t, :].T
    _bart_rev1 = _h_sim_bart[_n_t : 2 * _n_t, :].T
    _glm_rev0 = _h_sim_glm[:_n_t, :].T
    _glm_rev1 = _h_sim_glm[_n_t : 2 * _n_t, :].T

    _bart_hr = _bart_rev1 / np.clip(_bart_rev0, 1e-9, None)
    _glm_hr = _glm_rev1 / np.clip(_glm_rev0, 1e-9, None)
    _bart_med, (_bart_lo, _bart_hi) = (
        np.median(_bart_hr, axis=0),
        np.quantile(_bart_hr, [0.05, 0.95], axis=0),
    )
    _glm_med, (_glm_lo, _glm_hi) = (
        np.median(_glm_hr, axis=0),
        np.quantile(_glm_hr, [0.05, 0.95], axis=0),
    )
    _true_hr = 1.0 + sim_times / 36

    _fig_sim, _ax_sim = plt.subplots(figsize=(9, 5))
    _ax_sim.fill_between(
        sim_times, _bart_lo, _bart_hi, step="post", color="#4c72b0", alpha=0.2
    )
    _ax_sim.step(
        sim_times,
        _bart_med,
        where="post",
        color="#4c72b0",
        linewidth=2,
        label="BART median (90% CrI)",
    )
    _ax_sim.fill_between(
        sim_times, _glm_lo, _glm_hi, step="post", color="#c44e52", alpha=0.15
    )
    _ax_sim.step(
        sim_times,
        _glm_med,
        where="post",
        color="#c44e52",
        linewidth=2,
        linestyle="-",
        label="cloglog GLM median",
    )
    _ax_sim.plot(
        sim_times,
        _true_hr,
        color="black",
        linewidth=2,
        linestyle="--",
        label="True HR(t) = 1 + t/36",
    )
    _ax_sim.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    _ax_sim.set_xlabel("months since surgery")
    _ax_sim.set_ylabel("HR(t) = h(t | rev=1) / h(t | rev=0)")
    _ax_sim.set_title("Recovery of a non-proportional hazard from synthetic data")
    _ax_sim.legend(frameon=False, loc="upper left")
    _fig_sim.tight_layout()
    _fig_sim
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now to the real data

    With the machinery validated on a known DGP, we apply the same
    discrete-time hazard BART (and its cloglog GLM comparator) to the
    Roegele Tommy John surgery list.
    """)
    return


@app.cell
def _(RANDOM_SEED, pl):
    # Load Roegele's surgery list. We keep all professional levels and
    # all positions (not just MLB pitchers), so BART sees the full range
    # of recovery trajectories Roegele compiled. A 36-month
    # administrative horizon: anyone still not returned by then is
    # censored at t=36. The broader population is what makes Roegele's
    # descriptive table show the nonlinear age effect; the MLB-only
    # subset is too narrow to recover it. A handful of throws records
    # (R*, P, L/R) are dropped as data-entry artifacts.
    #
    # Deterministic subsample to 500 subjects keeps the person-month
    # expansion to roughly 12k rows and the live-tutorial BART fit to
    # roughly 6 minutes. Random_state fixed so attendees see the same
    # numbers we do.
    HORIZON = 36.0
    SUBSAMPLE_N = 500
    tj_df = (
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
        .sample(n=SUBSAMPLE_N, seed=RANDOM_SEED, shuffle=True)
    )
    f"{tj_df.height} subjects, {int(tj_df['event'].sum())} returns, {tj_df.height - int(tj_df['event'].sum())} censored at {HORIZON:.0f}-month horizon"
    return (tj_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A note on censoring vs competing events

    The discrete-time hazard model below treats every row with
    `event=0` as standard right-censoring: the subject was still at
    risk when our observation ended, and could in principle have
    returned later. That's the right interpretation for someone who
    is still in rehab as of the data pull.

    It's the wrong interpretation for a pitcher whose career ended
    without a return. Roegele's data dictionary classifies 64% of
    non-returns as exactly this — *competing events* (`active=0`
    with no return), not right-censoring. Treating career-enders as
    "still at risk" implicitly assumes they would have come back
    eventually, which biases the modelled survival curves upward.

    The 36-month administrative horizon helps the worst cases (anyone
    Roegele has lost track of for years) but does not address
    competing events inside the horizon. The honest fix is a
    competing-risks model. Sparapani et al. (2020, *SMMR* 29:57-77)
    extend BART to subdistribution hazards for exactly this setup;
    we treat that as out of scope for this notebook and flag the
    direction of bias instead.
    """)
    return


@app.cell
def _(np, pmb, tj_df):
    # Long-format expansion: each subject contributes one row per
    # integer month they remain at risk, ending in a final row in
    # which they either return (y=1) or are censored (y=0). The
    # discrete time index t goes in column 0 of X; BART can split on
    # it to learn the hazard's time shape jointly with covariate
    # effects rather than imposing a parametric baseline.
    #
    # Categorical encoding via sorted np.unique-style lookups: keep
    # the level ordering around so prediction profiles encode by name
    # (surgeon_levels.index("Andrews")) rather than hardcoding codes.
    surgeon_levels = sorted(tj_df["surgeon_group"].unique().to_list())
    throws_levels = sorted(tj_df["throws"].unique().to_list())
    surgeon_code = {s: i for i, s in enumerate(surgeon_levels)}
    throws_code = {th: i for i, th in enumerate(throws_levels)}

    _rows = []
    for _r in tj_df.iter_rows(named=True):
        # max(1, ceil(...)) guards against a same-day censor that
        # rounds to zero; today's filtered minimum is ~8 months so
        # this is purely defensive, but cheap.
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

    Two choices for the link function carry different assumptions.
    With a **complementary log-log** link,
    $$h(t \mid x) \;=\; 1 - \exp\!\bigl(-\exp(\alpha_t + x^\top \beta)\bigr),$$
    the discrete-time hazard ratio between any two profiles is exactly
    $\exp(\beta^\top (x_1 - x_2))$, constant over $t$. This is the
    discrete-time form of the Cox proportional hazards model. It is the
    only link in discrete time that yields strictly proportional hazards,
    which is the assumption Cox PH famously imposes.

    We give the baseline hazard $\alpha_t$ twelve degrees of freedom
    (one log-hazard per quarter, $t \in \{1\text{-}3, 4\text{-}6, \ldots, 34\text{-}36\}$),
    which is plenty of flexibility on the time axis. Any remaining
    disagreement with BART will then be honestly about the covariate
    effects, not the baseline.

    BART uses the same cloglog link as this GLM, so the classical-vs-BART
    contrast lives on a single axis: the function class of the linear
    predictor. Linear additive (GLM) vs. tree sum (BART). If $g(t, x)$
    were linear in this notebook's BART model, strict proportional
    hazards would hold there too; any time-varying hazard ratio we
    recover is unambiguously a consequence of the tree-sum function
    class, not the link.
    """)
    return


@app.cell
def _(np, surgeon_levels, surv_X):
    # Wide design matrix for the cloglog GLM comparator: standardized
    # continuous covariates, raw binary indicators, K-1 surgeon dummies
    # with the largest group (Andrews) as the reference category. The
    # quarter index q_idx is carried separately so the baseline log-hazard
    # alpha_q[q_idx] can be a 12-vector indexed at the row level rather
    # than a 12-column dummy block in the matrix.
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

    glm_feature_names = ["age_std", "year_std", "doy", "throws_R", "revision"] + [
        f"surg_{surgeon_levels[i]}" for i in _non_ref
    ]
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
    # Cloglog discrete-time hazard GLM: same Bernoulli likelihood on the
    # person-month expansion as the BART model, but the linear predictor
    # is alpha_q[q_idx] + X_glm @ beta and the link is cloglog. Each
    # beta_j is interpretable as a log hazard ratio that applies at every
    # t by construction. NUTS auto-registers since this model has no BART
    # RV.
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


@app.cell(hide_code=True)
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
    # Cloglog BART on the person-month expansion: each row asks "did
    # this subject return this month?". The hazard is
    # 1 - exp(-exp(eta)), so the link matches the GLM comparator and
    # the BART-vs-GLM contrast is purely about the function class on
    # eta (tree sum vs linear). The first column of X is the month
    # index t, so BART learns the hazard shape h(t, x) jointly with
    # covariate effects rather than imposing a parametric baseline.
    # split_rules wires the encoded categoricals to the right tree
    # geometry: continuous for t/age/year/doy, OneHot for binary
    # throws and revision, Subset for the 10-level surgeon group.
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
    # Reference profile holds covariates at their training-data modes /
    # medians for the broadened (all-levels, all-positions) population:
    # age 22, surgery year 2012, mid-season DOY, right-handed, first
    # surgery, Andrews. Each panel below sweeps one covariate while
    # pinning the rest at the reference values. Age contrast widened
    # to {20, 30, 40} now that the data spans 13-47.
    #
    # Two parallel encodings of every profile: X_profiles in the raw
    # 7-column BART design, and X_profiles_glm in the wide GLM design
    # (standardized continuous, surgeon dummies). q_idx_profiles maps
    # each row to its quarter for the alpha_q baseline.
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

    # Reference args for every profile, varying one covariate at a time.
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
    # GLM posterior predictive at the same profiles. The cloglog hazard
    # h_glm is a Deterministic of (alpha_q, beta), so var_names=["h_glm"]
    # returns posterior draws of the per-row hazard with no additional
    # Bernoulli sampling step. q_idx_data is re-attached alongside the
    # design matrix so alpha_q[q_idx] picks the right baseline.
    with model_glm:
        pm.set_data({"X_glm_data": X_profiles_glm, "q_idx_data": q_idx_profiles})
        pp_glm = pm.sample_posterior_predictive(
            idata_glm,
            var_names=["h_glm"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_glm,)


@app.cell
def _(np, pp_glm, pp_surv, profile_names, times):
    # .stack(sample=("chain","draw")) flattens chain x draw into a
    # single sample dimension. Each profile contributes len(times)
    # consecutive rows; slice them out, transpose so draws are the row
    # axis, then accumulate hazards via S(t) = prod_{s<=t}(1 - h(s)).
    # Build the same curves twice: from the BART posterior-predictive h
    # draws and from the GLM posterior-predictive h_glm draws.
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
    # Marginal Kaplan-Meier from the subject-level data: empirical
    # per-bin hazard events_t / at_risk_t, then S(t) = prod(1 - h).
    # No covariate adjustment. Serves as a no-covariate baseline that
    # each panel below overlays as a gray dashed reference; BART's
    # stratified curves should bracket it in the aggregate.
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

    The headline pitch for nonparametric survival models like BART is
    that they do **not** assume the hazard ratio between two covariate
    profiles is constant over time. The cloglog GLM does: by
    construction, the discrete-time cloglog model gives a hazard ratio
    of $\exp(\beta^\top (x_1 - x_2))$, the same at every $t$.

    The plot below contrasts the two on a single comparison: 40-year-old
    vs 20-year-old pitcher, all other covariates pinned at the
    reference. The GLM line is flat (its HR is `exp(beta_age * 20 /
    sd_age)`); BART's ribbon is allowed to be a curve. If the ribbon
    stays inside the GLM band, proportional hazards approximately holds
    for this contrast. If it drifts up or down as $t$ grows, BART has
    found a non-PH effect that Cox PH would miss.
    """)
    return


@app.cell(hide_code=True)
def _(age_sd_glm, h_curves, idata_glm, np, plt, times):
    _bart_hr_draws = h_curves["age_40"] / np.clip(h_curves["age_20"], 1e-9, None)
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
    _ax_hr.set_ylabel(r"HR$(t)$ = $h(t|\text{age}=40) / h(t|\text{age}=20)$")
    _ax_hr.set_title(
        f"Time-varying hazard ratio: 40 vs 20\ncloglog GLM HR = {_glm_hr_med:.2f} (90% CrI [{_glm_hr_lo:.2f}, {_glm_hr_hi:.2f}])"
    )
    _ax_hr.legend(frameon=False, loc="upper left")
    _fig_hr.tight_layout()
    _fig_hr
    return


@app.cell
def _(RANDOM_SEED, idata_surv, model_surv, pm, surv_X):
    # Full posterior-predictive event draws at the training rows.
    # Reattach X_data to surv_X (it was last set to X_profiles for the
    # profile-level PP). sample_vars=["eta"] is required for pymc-bart
    # under pm.sample_posterior_predictive.
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

    If the observed KM lands inside the cloud of simulated KMs, BART is
    well calibrated against the marginal survival distribution. A
    systematic deviation indicates structural mis-calibration that the
    contrast panels would not surface.
    """)
    return


@app.cell(hide_code=True)
def _(RANDOM_SEED, S_km, km_survival, np, plt, pp_train, times, tj_df):
    # Reconstruct per-subject row offsets from the same expansion rule
    # the data prep used (max(1, ceil(time_months)) periods per subject).
    _n_periods = np.maximum(1, np.ceil(tj_df["time_months"].to_numpy())).astype(int)
    _subject_starts = np.concatenate([[0], np.cumsum(_n_periods)])
    _n_subjects = len(_n_periods)

    # Stack chain/draw, pick 100 random columns, integer-typed for speed.
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
    Pre-summing over `t` lets us print BART and the cloglog GLM side by
    side with 90% credible intervals, so the disagreements between the
    two models become numerical rather than purely visual.
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

    Variable importance for BART comes in two flavours. **Forward**
    inclusion ("VI", the default) counts how often each variable is used
    in the posterior trees. **Backward** elimination instead refits
    restricted submodels by dropping features one at a time and reports
    $R^2$ against the full model as features come back in. Forward is
    fast and good at ranking; backward is slower but more honest about
    correlated features that the forward count may double-credit.

    The seven columns are time `t`, `age`, `surgery_year`,
    `surgery_doy_frac`, `throws`, `revision`, and `surgeon_group`. We
    expect `t` to dominate either way (the hazard is highly time-shaped);
    secondary ranks may differ between methods if any two covariates
    carry correlated signal.
    """)
    return


@app.cell(hide_code=True)
def _(
    RANDOM_SEED,
    eta,
    feature_names,
    idata_surv,
    model_surv,
    plt,
    pm,
    pmb,
    surv_X,
):
    with model_surv:
        pm.set_data({"X_data": surv_X})
        _vi_fwd = pmb.compute_variable_importance(
            idata_surv, eta, surv_X, method="VI", random_seed=RANDOM_SEED
        )
        _vi_back = pmb.compute_variable_importance(
            idata_surv, eta, surv_X, method="backward", random_seed=RANDOM_SEED
        )

    _fig_vi, _axes_vi = plt.subplots(1, 2, figsize=(13, 4.5))
    pmb.plot_variable_importance(_vi_fwd, labels=feature_names, ax=_axes_vi[0])
    pmb.plot_variable_importance(_vi_back, labels=feature_names, ax=_axes_vi[1])
    _axes_vi[0].set_title("Forward (inclusion-frequency)")
    _axes_vi[1].set_title("Backward (elimination)")
    _fig_vi.tight_layout()
    _fig_vi
    return


@app.cell(hide_code=True)
def _(eta, model_surv, pm, pmb, surv_X, surv_y):
    # var_discrete=[0, 4, 5, 6]: time t, throws, revision, surgeon
    # are integer-coded so PDP renders them as ticks rather than
    # smooth curves. age, surgery_year, surgery_doy_frac stay
    # continuous.
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


@app.cell(hide_code=True)
def _(RANDOM_SEED, eta, model_surv, pm, pmb, surv_X, surv_y):
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
def _(az, idata_surv):
    az.plot_convergence_dist(idata_surv, var_names=["eta"])
    return


@app.cell
def _(idata_surv):
    _stats = idata_surv.sample_stats
    (
        f"divergences: {int(_stats['diverging'].sum())}"
        if "diverging" in _stats
        else "PGBART-only model (no HMC step); divergence diagnostic not applicable"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
