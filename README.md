# Flexible Statistical Modeling with Bayesian Additive Regression Trees

### PyData London 2026 | 90-minute tutorial 

Most machine learning methods give you a prediction but not a measure of how much to trust it. Bayesian Additive Regression Trees (BART) combine the flexibility of tree ensembles (e.g. random forests, boosting) with full uncertainty quantification—every prediction comes with a probability interval, not just a point estimate. This hands-on tutorial introduces BART for regression and classification. Using `pymc-bart`, participants will learn to fit flexible models that automatically capture non-linear relationships while providing honest uncertainty estimates. We emphasize practical interpretation throughout: visualizing predictions with uncertainty bands, understanding variable importance, and interpreting model output.


## Prior Knowledge Expected

Basic familiarity with Python and the basic scientific stack. Undergraduate-level statistics. No prior experience with BART or tree-based methods is assumed.


## Outline

The tutorial is delivered as a slide deck plus three marimo notebooks: a PyMC
introduction and two hands-on application notebooks:

- **How BART works (25 min, slide presentation):** Bayesian inference and MCMC primer; sum-of-weak-trees; regularization priors that prevent overfitting; posterior inference over tree structures; why this gives calibrated uncertainty. Delivered as slides rather than a notebook; the figures are built from a from-scratch implementation in pure NumPy.
- **`pymc_intro.py` — Building models with PyMC (20 min):** Introduce PyTensor's symbolic graph, PyMC model contexts and random variables, prior and observed variables, deterministic LD50 summaries, prior predictive checks, posterior sampling, and posterior predictive sampling with `pm.Data`.
- **`regression.py` — BART for regression (25 min, hands-on):** Fit a first `pymc-bart` model on 2024 Formula 1 lap times, check posterior predictive uncertainty on held-out laps, interpret variable importance and partial dependence, then use the diagnostics to motivate a richer F1 model and a tree-count comparison.
- **`classification.py` — BART for classification (15 min, hands-on):** Predict whether GSS respondents report very high life satisfaction, check held-out probabilities and reliability, compare against logistic regression, then return to the full satisfaction ladder with an ordered-probit BART model.
- **Wrap-up (5 min):** When to use BART vs. alternatives; resources for further learning


## Running the notebooks

```bash
pixi install                                      # first-time setup
pixi run marimo edit regression.py                 # author / teach in the browser
pixi run marimo check pymc_intro.py regression.py classification.py  # structural validation
```

## Notebook writing conventions

- Put explanation in markdown cells; keep code comments local to implementation details.
- Use plain teaching transitions instead of workflow labels.
- Keep each notebook moving through the same arc: introduce the problem, fit, check, interpret, extend.

## Description

Machine learning models are often evaluated on predictive accuracy alone, but accuracy without uncertainty can be misleading. Classical tree ensemble methods like random forests and gradient boosting provide point predictions, and while techniques like conformal inference or bootstrap aggregation can add uncertainty estimates, these are often poorly calibrated or computationally expensive.

Bayesian Additive Regression Trees (BART) offer a different approach: uncertainty quantification is built into the model, not ignored or bolted on afterward. BART models the response as a sum of small trees, with regularization priors that keep each tree weak. Posterior inference over the tree structures yields a full distribution over predictions—every fitted value comes with a credible interval that reflects genuine uncertainty about the underlying function.

Before the BART applications, `pymc_intro.py` introduces the PyMC model-building API with the rat-toxicity example: model contexts, named dimensions, priors, deterministic variables, observed variables, prior predictive checks, posterior sampling, and posterior prediction.

This tutorial introduces BART through two applications, each demonstrating how uncertainty changes the way we interpret results:

**Regression:** We begin with continuous outcomes, fitting BART models and visualizing posterior predictive distributions. Rather than a single fitted curve, participants will see HDI bands that widen where data is sparse and narrow where evidence is strong. We'll explore variable importance—which comes with its own uncertainty—and partial dependence plots that reveal non-linear effects.

**Classification:** For binary outcomes, BART produces predicted probabilities with uncertainty, not just class labels. We'll examine how this uncertainty propagates through decision-making and compare calibration against a standard logistic-regression classifier.


## Keywords

Bayesian inference, BART, uncertainty quantification, machine learning, regression, classification, PyMC
