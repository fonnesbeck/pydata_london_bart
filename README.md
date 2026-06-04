# Flexible Statistical Modeling with Bayesian Additive Regression Trees

### PyData London 2026 | 90-minute tutorial 

Most machine learning methods give you a prediction but not a measure of how much to trust it. Bayesian Additive Regression Trees (BART) combine the flexibility of tree ensembles (e.g. random forests, boosting) with full uncertainty quantification—every prediction comes with a probability interval, not just a point estimate. This hands-on tutorial introduces BART through three applications: regression, classification, and survival analysis. Using `pymc-bart`, participants will learn to fit flexible models that automatically capture non-linear relationships while providing honest uncertainty estimates. We emphasize practical interpretation throughout: visualizing predictions with uncertainty bands, understanding variable importance, and interpreting model output.


## Prior Knowledge Expected

Basic familiarity with Python and the basic scientific stack. Undergraduate-level statistics. Familiarity with survival analysis concepts (censoring, hazard functions) is helpful but not required. No prior experience with BART or tree-based methods is assumed.


## Outline

The tutorial is delivered as one slide-source notebook plus three hands-on
application notebooks:

- **`slides/how_bart_works.py` — How BART works (25 min):** Bayesian inference and MCMC primer; sum-of-weak-trees; regularization priors that prevent overfitting; posterior inference over tree structures; why this gives calibrated uncertainty. Built bottom-up in pure NumPy so the sampler reads top-to-bottom.
- **Break (5 min)**
- **`01_regression.py` — BART for regression (25 min, hands-on):** Live path: fit a first `pymc-bart` model, visualize posterior predictive uncertainty, check diagnostics, and read variable-importance / partial-dependence plots. Optional / pre-run sections cover model escalation and tree-count selection. 2024 Formula 1 lap times across five Grands Prix (Bahrain, Monaco, Spain, Britain, Monza).
- **`02_classification.py` — BART for classification (15 min, hands-on):** **Live path:** predicted probabilities with uncertainty (not just class labels), reliability, and a logistic-regression baseline on the same held-out split. **Take-home extension:** ordered probit on the full life-satisfaction scale.
- **`03_survival.py` — BART for survival analysis (15 min, hands-on):** **Live path:** a short survival primer, discrete-time hazards, survival curves with credible intervals, and individualized risk assessment. **Optional / pre-run:** synthetic checks and proportional-hazards diagnostics. **Take-home extension:** variable-importance and PDP/ICE follow-ups.
- **Wrap-up (5 min):** When to use BART vs. alternatives; resources for further learning


## Running the notebooks

```bash
pixi install                                      # first-time setup
pixi run marimo edit 01_regression.py              # author / teach in the browser
pixi run marimo check 0[1-3]_*.py slides/how_bart_works.py  # structural validation
```


## Description

Machine learning models are often evaluated on predictive accuracy alone, but accuracy without uncertainty can be misleading. Classical tree ensemble methods like random forests and gradient boosting provide point predictions, and while techniques like conformal inference or bootstrap aggregation can add uncertainty estimates, these are often poorly calibrated or computationally expensive.

Bayesian Additive Regression Trees (BART) offer a different approach: uncertainty quantification is built into the model, not ignored or bolted on afterward. BART models the response as a sum of small trees, with regularization priors that keep each tree weak. Posterior inference over the tree structures yields a full distribution over predictions—every fitted value comes with a credible interval that reflects genuine uncertainty about the underlying function.

This tutorial introduces BART through three applications, each demonstrating how uncertainty changes the way we interpret results:

**Regression:** We begin with continuous outcomes, fitting BART models and visualizing posterior predictive distributions. Rather than a single fitted curve, participants will see HDI bands that widen where data is sparse and narrow where evidence is strong. We'll explore variable importance—which comes with its own uncertainty—and partial dependence plots that reveal non-linear effects.

**Classification:** For binary outcomes, BART produces predicted probabilities with uncertainty, not just class labels. We'll examine how this uncertainty propagates through decision-making and compare calibration against a standard logistic-regression classifier.

**Survival analysis:** Time-to-event data is inherently uncertain, and BART's flexibility is particularly valuable when the hazard function has unknown shape. Participants will fit survival models and plot individualized survival curves with credible intervals—essential for communicating risk to stakeholders.


## Keywords

Bayesian inference, BART, uncertainty quantification, machine learning, regression, classification, survival analysis, PyMC
