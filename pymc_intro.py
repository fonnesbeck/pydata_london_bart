import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import pymc as pm
    import arviz_plots as azp

    RANDOM_SEED = 42

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"

    pymc_template = go.layout.Template()
    pymc_template.layout = go.Layout(
        colorway=[
            PYMC_BLUE,
            PYMC_GREEN,
            PYMC_LIGHT_BLUE,
            "#15726C",
            "#C2C240",
            PYMC_DARK_GREEN,
            "#151B72",
            "#40C240",
        ],
        font=dict(color="#333"),
        title=dict(font=dict(color=PYMC_BLUE)),
    )
    pio.templates["pymc_labs"] = pymc_template
    pio.templates.default = "plotly_white+pymc_labs"
    return RANDOM_SEED, azp, go, np, pm


@app.cell(hide_code=True)
def header(mo):
    mo.md(r"""
    # Introduction to PyMC

    A short introduction for users who have not used PyMC before: model contexts, random variables, priors, deterministic quantities, observed data, posterior sampling, and posterior predictive prediction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Contexts and Random Variables

    The canonical way to specify PyMC models is using a `Model` context manager. Generally speaking, a context manager is a Python idiom that defines what happens when entering and exiting a `with` statement. They provide a clean, reliable way to set up and tear down resources.

    As an analogy, `Model` is a tape machine that records what is being added to the model; it keeps track of the random variables (observed or unobserved) and other model components.
    """)
    return


@app.cell
def _(pm):
    with pm.Model() as intro_model:
        pm.Normal("z", mu=0.0, sigma=5.0)
    intro_model
    return (intro_model,)


@app.cell
def _(intro_model):
    intro_model.named_vars
    return


@app.cell
def _(intro_model):
    intro_model.compile_logp()({"z": 2.5})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Core Components

    - **Stochastic Random Variables**: Variables whose values are not completely determined by their parents. These represent uncertainty in the model parameters or data generating process.
        - **Prior distributions** for model parameters
        - **Likelihood distributions** for observed data

        $$x \sim \text{Normal}(\mu, \sigma)$$

    - **Deterministic Variables**: Variables whose values are completely determined by their parents through a mathematical operation; no additional randomness is added by the operation. These represent transformations or combinations of other variables.

        $$y = \mu + \beta x$$

    These building blocks connect in a directed acyclic graph that completely specifies a joint probability distribution. PyMC leverages this graph structure to efficiently sample from the posterior distribution using its available inference algorithms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distributions, shape, and dims

    A stochastic variable is represented in PyMC by a distribution class. Distribution subclasses accept several arguments when constructed. Some of the most important are:

    `name`
    : Name for the new model variable. This argument is **required**, and is used as a label and index value for the variable.

    `shape`
    : The variable's shape.

    `dims`
    : A tuple of dimension names known to the model.
    """)
    return


@app.cell
def _(pm):
    with pm.Model(coords={"city": ["Vancouver", "Calgary", "Toronto", "Montreal", "Halifax"]}) as dims_model:
        pm.Normal("x_city", dims="city")

    dims_model
    return (dims_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Users call the `pm.draw()` function to simulate values from a random variable.
    """)
    return


@app.cell
def _(dims_model, pm):
    x_city = dims_model["x_city"]
    pm.draw(x_city, draws=5, random_seed=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building Models in PyMC

    Now that we understand the basic building blocks of PyMC models, let's see how to combine them to build a complete model.

    Recall our dataset:
    """)
    return


@app.cell
def _(np):
    dose = np.array([-0.86, -0.3, -0.05, 0.73])
    n = 5
    deaths = np.array([0, 1, 3, 5])
    return deaths, dose, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this dataset `log_dose` includes 4 levels of dosage, on the log scale, each administered to `n=5` rats during the experiment. The response variable is `deaths`, the number of positive responses to the dosage.

    The number of deaths can be modeled as a binomial response, with the probability of death being a linear function of dose:

    $$
    \begin{aligned}
    \text{logit}(p_i) &= \beta_0 + \beta_1 x_i \\
    y_i &\sim \text{Bin}(n_i, p_i) \\
    \end{aligned}
    $$

    Our research interest is in estimating the **LD50**, the dose at which 50% of the rats die.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior Specification

    The first step in specifying a PyMC model is defining the prior distributions for each unknown parameter.

    - what are the unknown parameters?
    - which distributions should we use to characterize our beliefs about their values?

    We need to create priors for the regression coefficients. We will use a positive-valued distribution for the slope.
    """)
    return


@app.cell
def _(deaths, dose, n, pm):
    def build_dose_model():
        with pm.Model() as model:
            beta0 = pm.Normal("beta0", 0, sigma=1)
            beta1 = pm.Lognormal("beta1", 0, sigma=1)
            p = pm.math.invlogit(beta0 + beta1 * dose)
            pm.Deterministic("ld50", -beta0 / beta1)
            pm.Binomial("y", n=n, p=p, observed=deaths)
        return model

    dose_model = build_dose_model()
    dose_model
    return (dose_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deterministic Variables

    A deterministic variable is one whose values are **completely determined** by the values of their parents.

    In our model, the probability of death is a deterministic function of the dose and the parameters of the model.

    To ensure that deterministic variables' values are accumulated during sampling, they should be instantiated using the **named deterministic** interface; this uses the `Deterministic` function to create the variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observed Random Variables

    Stochastic random variables whose values are observed are represented by a different class than unobserved random variables. An observed random variable is instantiated any time a stochastic variable is specified with data passed as the `observed` argument.

    In our model, the observed rat deaths are represented by an observed random variable using a binomial random variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimating the model

    If we are happy with our model specification, we can go ahead and estimate the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see what our simulated datasets look like:
    """)
    return


@app.cell
def _(RANDOM_SEED, dose_model, pm):
    with dose_model:
        trace = pm.sample(random_seed=RANDOM_SEED)
    trace
    return (trace,)


@app.cell
def _(azp, trace):
    azp.plot_trace_dist(trace, var_names=["beta0", "beta1", "ld50"])
    return


@app.cell(hide_code=True)
def _(deaths, dose, go, n, np, trace):
    def plot_dose_response_posterior():
        empirical_p = deaths / n
        dose_range = np.linspace(dose.min(), dose.max(), 100)
        beta0_samples = trace.posterior["beta0"].values.flatten()
        beta1_samples = trace.posterior["beta1"].values.flatten()

        def logistic(x):
            return 1 / (1 + np.exp(-x))

        fig = go.Figure()
        for i in np.arange(-250, 0):
            prob = logistic(beta0_samples[i] + beta1_samples[i] * dose_range)
            fig.add_trace(
                go.Scatter(
                    x=dose_range,
                    y=prob,
                    mode="lines",
                    line=dict(color="blue", width=1),
                    opacity=0.1,
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=dose,
                y=empirical_p,
                mode="markers",
                marker=dict(color="black", size=8),
                showlegend=False,
            )
        ).update_layout(
            xaxis_title="log(dose)",
            yaxis_title="Probability of death",
            template="simple_white",
            showlegend=False,
        )
        return fig

    plot_dose_response_posterior()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction and forecasting

    We might also be interested in predicting on unseen data. For example, how many deaths would we expect if we administered the LD50 dose?

    In PyMC, we can use `pm.Data` objects for our data. It allows you to define data as a symbolic node in the model that you can later switch out for other data.
    """)
    return


@app.cell
def _(deaths, dose, pm):
    with pm.Model(coords=dict(coeffs=["intercept", "slope"], doses=["first", "second", "third", "fourth"])) as prediction_model:
        dose_data = pm.Data("dose_data", dose, dims="doses")
        deaths_data = pm.Data("deaths_data", deaths, dims="doses")
    return deaths_data, dose_data, prediction_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the `coords` argument in the `pm.Data` constructor. This allows us to specify named dimensions for the data variable, which is useful when working with multi-dimensional data.

    In the previous model, we specified separate variables for the slope and intercept. We can instead use a single vector-valued variable with named dimensions to represent both coefficients.
    """)
    return


@app.cell
def _(pm, prediction_model):
    with prediction_model:
        beta = pm.Normal("beta", 0, sigma=2, dims="coeffs")
    return (beta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rest of the model remains the same:
    """)
    return


@app.cell
def _(beta, deaths_data, dose_data, n, pm, prediction_model):
    with prediction_model:
        p_prediction = pm.Deterministic(
            "p", pm.math.invlogit(beta[0] + beta[1] * dose_data), dims="doses"
        )
        pm.Deterministic("ld50", -beta[0] / beta[1])
        pm.Binomial("y", n=n, p=p_prediction, observed=deaths_data, dims="doses")
    prediction_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we estimate the model:
    """)
    return


@app.cell
def _(RANDOM_SEED, pm, prediction_model):
    with prediction_model:
        prediction_trace = pm.sample(random_seed=RANDOM_SEED)
    prediction_trace
    return (prediction_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For our prediction, we want to pass the LD50 dose to the model. We will just take the mean of the posterior samples for the LD50.
    """)
    return


@app.cell
def _(prediction_trace):
    prediction_trace.posterior["ld50"].values.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Swapping data is done with `pm.set_data`. Then, we sample from the **posterior predictive distribution** by calling `pm.sample_posterior_predictive`. This samples data from the model using the fitted parameters and the new data.

    The zero passed to `deaths_data` is just a dummy value, since we are not conditioning on any observed deaths.
    """)
    return


@app.cell
def _(pm, prediction_model, prediction_trace):
    with prediction_model:
        pm.set_data(
            {"dose_data": [prediction_trace.posterior["ld50"].values.mean()], "deaths_data": [0]},
            coords={"doses": ["ld50"]},
        )
        predictive_samples = pm.sample_posterior_predictive(prediction_trace)
    predictive_samples
    return (predictive_samples,)


@app.cell(hide_code=True)
def _(go, n, np, predictive_samples):
    def plot_ld50_predictive():
        predicted_deaths = predictive_samples.posterior_predictive["y"].values.flatten()
        counts = np.bincount(predicted_deaths.astype(int), minlength=n + 1)
        probs = counts / counts.sum()
        fig = go.Figure(go.Bar(x=list(range(n + 1)), y=probs))
        fig.update_layout(
            xaxis_title="Number of Deaths (out of 5)",
            yaxis_title="Probability",
            title="Posterior Predictive: Deaths at LD50 Dose",
            xaxis=dict(dtick=1),
        )
        return fig

    plot_ld50_predictive()
    return


if __name__ == "__main__":
    app.run()
