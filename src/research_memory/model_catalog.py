"""Method vocabulary and the narrow methods this pilot Runner can execute."""

DESIGNS = {
    "descriptive": "Descriptive / correlation",
    "observational_association": "Observational association",
    "randomized_experiment": "Randomized experiment",
    "panel_fixed_effects": "Panel / fixed effects",
    "difference_in_differences": "Difference-in-Differences",
    "event_study": "Event study",
    "instrumental_variables": "Instrumental variables / 2SLS",
    "regression_discontinuity": "Regression discontinuity",
    "synthetic_control": "Synthetic control",
    "matching_weighting": "Matching / weighting",
    "survival_duration": "Survival / duration",
    "network_dyadic": "Network / dyadic model",
    "structural_model": "Structural estimation",
    "sem_latent_variable": "SEM / latent-variable model",
    "predictive_ml": "Prediction / machine learning",
}

OUTCOMES = {
    "continuous": "Continuous",
    "binary": "Binary",
    "count": "Count",
    "fractional": "Fraction / proportion",
    "ordinal": "Ordinal",
    "multinomial": "Unordered categories",
    "time_to_event": "Time to event",
    "dyadic": "Pair / network relation",
    "text": "Text-derived outcome",
}

ESTIMATORS = {
    "OLS", "fixed_effects_OLS", "logistic_regression", "probit",
    "fixed_effects_logistic_regression", "conditional_logistic_regression",
    "poisson", "negative_binomial", "hurdle", "zero_inflated_count",
    "fractional_logit", "beta_regression", "ordered_logit", "ordered_probit",
    "multinomial_logit", "Cox", "parametric_duration", "IV_2SLS",
    "difference_in_differences", "event_study", "local_polynomial_RDD",
    "synthetic_control", "matching_or_weighting", "network_model",
    "structural_estimation", "SEM", "other",
}

# These labels are executable only for the current tagged-outcome pilot runner.
RUNNER_ESTIMATORS = {
    "OLS", "fixed_effects_OLS", "logistic_regression",
    "fixed_effects_logistic_regression", "conditional_logistic_regression",
}

DESIGN_REQUIREMENTS = {
    "difference_in_differences": "treatment group, treatment timing, repeated pre/post outcomes, parallel-trends justification",
    "event_study": "event timing, event-time window/reference period, and anticipation/pre-trend checks",
    "instrumental_variables": "endogenous treatment, instrument(s), relevance, exclusion/exogeneity, and target estimand",
    "regression_discontinuity": "running variable, cutoff, assignment rule, bandwidth, and continuity/no-manipulation checks",
    "synthetic_control": "treated unit, donor pool, long pre-treatment outcome history, and placebo checks",
    "matching_weighting": "treatment assignment, covariates, overlap, and conditional-exchangeability justification",
    "survival_duration": "time origin, event indicator, censoring rule, and duration distribution/hazard choice",
    "panel_fixed_effects": "panel unit/time identifiers and within-unit source of variation",
    "randomized_experiment": "randomization unit, assignment mechanism, treatment/comparison arms, and assignment-level inference",
    "network_dyadic": "node/edge definition, network boundary, dependence structure, and network-generating assumptions",
    "structural_model": "economic primitives, equilibrium/behavioral restrictions, and estimation/identification strategy",
    "sem_latent_variable": "measurement model, construct indicators, structural paths, and model-fit criteria",
    "predictive_ml": "prediction target/horizon, train-validation split, leakage controls, and held-out metric",
}

# Fixed data-dictionary facts for the current product-tagging pilot; no raw-data scan.
PILOT_DATA_FIELDS = frozenset({"tagged", "choice_set", "user_label", "typicality", "selfclaim_osa",
    "expertise_style", "expertise_genre", "informative", "distinctive", "previous_tags",
    "exemplarity", "selfclaim_lv", "selfclaim_cos", "selfclaim_jw",
    "subsample_6m", "subsample_1m", "subsample_1w", "subsample_q75", "subsample_q50",
    "subsample_q25", "subsample_top"})
DATA_REQUIRED_FIELDS = {
    "randomized_experiment": {"treatment_assignment"},
    "panel_fixed_effects": {"panel_unit", "time_variable"},
    "difference_in_differences": {"treatment_assignment", "treatment_timing", "time_variable"},
    "event_study": {"treatment_timing", "time_variable"},
    "instrumental_variables": {"endogenous_treatment", "instrument"},
    "regression_discontinuity": {"running_variable", "cutoff"},
    "synthetic_control": {"treated_unit", "donor_pool", "time_variable"},
    "matching_weighting": {"treatment_assignment"},
    "survival_duration": {"event_indicator", "duration"},
    "network_dyadic": {"network_edges"},
}


def data_gaps(design):
    """Return design inputs absent from the reviewed pilot data dictionary."""
    return sorted(DATA_REQUIRED_FIELDS.get(design, set()) - PILOT_DATA_FIELDS)


PILOT_ANALYSIS_GUIDANCE = (
    "This dataset supports the associational version of the question: binary tagged outcome, numeric "
    "predictors, choice_set groups and user_label clusters. Treatment assignment, panel time, instruments, "
    "RD cutoff, event duration and network edges are not documented; make no causal claim or select a "
    "causal/panel design. Observational association is available, so provide its exact Runner action; do "
    "not return a null action. Start with the declared focal predictor and no controls. Add at most two "
    "only when theory requires them; do not add every available predictor. The focal predictor is not a "
    "treatment. With linear/logit plus choice_set, report choice_set under fixed_effects and leave strata "
    "empty; conditional_logit uses strata and no fixed effect. Never report the same grouping as both. "
    "Use JSON null for absent model fields and 'none' only where the action schema offers it. "
    "Give one short rationale sentence."
)


def runner_status(specification, action):
    """Return an explicit execution capability verdict; never relabel unsupported methods."""
    if specification.get("design_type") != "observational_association":
        design = specification.get("design_type", "unknown")
        gaps = data_gaps(design)
        if gaps:
            required = DESIGN_REQUIREMENTS.get(design)
            reason = "Data dictionary lacks: " + ", ".join(gaps)
            return {"status": "data_insufficient", "reason": reason + ("; design also needs " + required if required else "")}
        return {"status": "unsupported", "reason": DESIGN_REQUIREMENTS.get(
            design, "This research design is not implemented by the current pilot Runner.")}
    if specification.get("estimator") not in RUNNER_ESTIMATORS:
        return {"status": "unsupported", "reason": "Estimator is not implemented by the current pilot Runner."}
    if specification.get("outcome_type") != "binary":
        return {"status": "unsupported", "reason": "Current Runner is bound to the binary tagged outcome."}
    if not action:
        return {"status": "action_required", "reason": "An exact Runner action is required to verify the outcome and terms."}
    if action.get("outcome") != "tagged":
        return {"status": "unsupported", "reason": "Current Runner is bound to the binary tagged outcome."}
    return {"status": "supported", "reason": "Exact action is supported by the current pilot Runner."}
