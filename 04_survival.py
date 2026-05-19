import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # BART for survival analysis

        Time-to-event data is inherently uncertain, and BART's flexibility
        is particularly valuable when the hazard function has unknown
        shape. We model **discrete-time hazards**: at each integer time $t$
        the event occurs with probability
        $$h(t \mid x) = \Phi\bigl(g(t, x)\bigr),$$
        where $g$ is a sum of trees and $\Phi$ is the probit link. Each
        subject contributes one row per time period they're at risk; the
        outcome is "did the event occur in this period?" — exactly the
        binary classification setup from notebook 3, applied to expanded
        person-time data.

        Survival follows from the hazards:
        $$S(t \mid x) = \prod_{s \le t} \bigl(1 - h(s \mid x)\bigr).$$
        """
    )
    return


@app.cell
def _():
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pymc as pm
    import pymc_bart as pmb

    RANDOM_SEED = 20260608
    rng = np.random.default_rng(RANDOM_SEED)
    return RANDOM_SEED, az, np, plt, pm, pmb, rng


@app.cell
def _(np, rng):
    def simulate_survival(n, rng, max_time=12):
        x1 = rng.uniform(-1, 1, size=n)
        x2 = rng.uniform(-1, 1, size=n)
        rows_y, rows_t, rows_x1, rows_x2 = [], [], [], []
        for i in range(n):
            for t in range(1, max_time + 1):
                logit_h = -3.0 + 0.7 * x1[i] + 0.4 * x2[i] * np.log(t)
                p = 1 / (1 + np.exp(-logit_h))
                y = rng.binomial(1, p)
                rows_y.append(y)
                rows_t.append(t)
                rows_x1.append(x1[i])
                rows_x2.append(x2[i])
                if y == 1:
                    break
        return (
            np.array(rows_y, dtype=int),
            np.column_stack([rows_t, rows_x1, rows_x2]).astype(float),
        )

    surv_y, surv_X = simulate_survival(150, rng, max_time=12)
    f"{surv_X.shape[0]} person-time rows, {surv_y.sum()} events"
    return surv_X, surv_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Configuring the PGBART step

        `pm.sample` auto-registers `pmb.PGBART` as the step method for any
        BART random variable in the model, but the sampler is configurable
        when the defaults don't suit the problem. Two kwargs matter most:

        - **`num_particles`** (default `10`): number of particles used in
          the conditional sequential Monte Carlo proposal. More particles
          give better proposals at higher per-step cost. The fit below
          uses `num_particles=20`.
        - **`batch`** (default `(0.1, 0.1)`): a `(tune_fraction,
          post_tune_fraction)` pair giving the fraction of the `m` trees
          refit per Gibbs sweep. Higher fractions move faster through tree
          space but each step costs more. The fit below uses
          `(0.1, 0.15)`, refitting slightly more trees once tuning ends.

        The fit cell below passes both knobs through an explicit
        `step=pmb.PGBART(vars=[eta], num_particles=20, batch=(0.1, 0.15))`.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, pm, pmb, surv_X, surv_y):
    # Discrete-time hazards modelled as repeated probit BART. The first
    # column of X is time t; BART learns h(t, x) jointly so the hazard
    # shape doesn't have to be specified up front.
    with pm.Model() as model_surv:
        X_data = pm.Data("X_data", surv_X)
        eta = pmb.BART("eta", X=X_data, Y=surv_y.astype(float), m=100)
        p = pm.Deterministic("p", pm.math.invprobit(eta))
        pm.Bernoulli("event", p=p, observed=surv_y, shape=p.shape)
        idata_surv = pm.sample(
            random_seed=RANDOM_SEED,
            step=pmb.PGBART(vars=[eta], num_particles=20, batch=(0.1, 0.15)),
        )
    return eta, idata_surv, model_surv


@app.cell
def _(np):
    # Predict hazards at two contrasting risk profiles, then accumulate
    # into survival curves.
    times = np.arange(1, 13)
    profile_low = np.column_stack([times, np.full(12, -0.8), np.full(12, 0.0)])
    profile_high = np.column_stack([times, np.full(12, 0.8), np.full(12, 0.0)])
    X_profiles = np.concatenate([profile_low, profile_high], axis=0)
    return X_profiles, times


@app.cell
def _(RANDOM_SEED, X_profiles, idata_surv, model_surv, pm):
    with model_surv:
        pm.set_data({"X_data": X_profiles})
        pp_surv = pm.sample_posterior_predictive(
            idata_surv,
            var_names=["p"],
            sample_vars=["eta"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_surv,)


@app.cell
def _(np, pp_surv):
    _p_draws = pp_surv.predictions["p"].stack(sample=("chain", "draw")).values
    p_low_draws = _p_draws[:12, :].T
    p_high_draws = _p_draws[12:, :].T
    S_low = np.cumprod(1 - p_low_draws, axis=1)
    S_high = np.cumprod(1 - p_high_draws, axis=1)
    return S_high, S_low


@app.cell(hide_code=True)
def _(S_high, S_low, np, plt, times):
    _m_low = S_low.mean(axis=0)
    _lo_low, _hi_low = np.quantile(S_low, [0.05, 0.95], axis=0)
    _m_high = S_high.mean(axis=0)
    _lo_high, _hi_high = np.quantile(S_high, [0.05, 0.95], axis=0)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.step(times, _m_low, where="post", color="#4c72b0", label=r"$x_1 = -0.8$")
    _ax.fill_between(times, _lo_low, _hi_low, step="post", color="#4c72b0", alpha=0.25)
    _ax.step(times, _m_high, where="post", color="#c44e52", label=r"$x_1 = +0.8$")
    _ax.fill_between(
        times, _lo_high, _hi_high, step="post", color="#c44e52", alpha=0.25
    )
    _ax.set_ylim(0, 1.02)
    _ax.set_xlabel("time $t$")
    _ax.set_ylabel(r"$S(t \mid x)$")
    _ax.set_title("Predicted survival, posterior mean and 90% band")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Which covariates drive the hazard?

        Variable importance and partial dependence work the same way for
        the discrete-time hazard model as for regression: the BART RV
        `eta` is the latent linear predictor of the probit hazard, and
        the helper functions ask which inputs influence it. The three
        columns are time $t$, $x_1$, $x_2$. The DGP made $x_1$ a
        constant-effect risk factor and $x_2$ a time-varying one, so we
        expect non-trivial importance for both.
        """
    )
    return


@app.cell(hide_code=True)
def _(eta, idata_surv, model_surv, pm, pmb, surv_X):
    with model_surv:
        pm.set_data({"X_data": surv_X})
        _vi_surv = pmb.compute_variable_importance(idata_surv, eta, surv_X)
    pmb.plot_variable_importance(_vi_surv, labels=["t", "x1", "x2"])
    return


@app.cell(hide_code=True)
def _(eta, model_surv, pm, pmb, surv_X, surv_y):
    with model_surv:
        pm.set_data({"X_data": surv_X})
        pmb.plot_pdp(
            bartrv=eta,
            X=surv_X,
            Y=surv_y.astype(float),
            var_discrete=[0],
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### From PDP to ICE: individual conditional expectations

        The partial dependence plots above marginalize each covariate's
        effect *across* the person-time rows: one curve per variable,
        averaged over all individuals. That is the right view for a global
        "does $x_1$ matter?" question, but it can hide heterogeneous
        effects — exactly the regime where individualized risk matters,
        which the README explicitly frames the survival section around.

        **Individual conditional expectation (ICE)** plots address this
        directly: one curve per *instance*, so when the effect of a
        covariate varies across the population, you see a fan of curves
        instead of a single average.

        The DGP for this notebook makes the contrast concrete:
        $$\operatorname{logit} h(t \mid x) = -3 + 0.7\, x_1 + 0.4\, x_2 \log t.$$
        The $x_1$ effect is constant across individuals (parallel curves);
        the $x_2$ effect is scaled by $\log t$, so individuals at later
        times respond more strongly to $x_2$ than individuals at $t=1$ —
        which on an ICE plot shows up as a spread of slopes that the PDP
        average smooths out.
        """
    )
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
        else "PGBART-only model (no HMC step) — divergence diagnostic not applicable"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
