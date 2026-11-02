"""Generate a bounded R analysis from a structured action; never execute it here."""
import re


PREDICTORS = frozenset((
    "typicality", "selfclaim_osa", "expertise_style", "expertise_genre",
    "informative", "distinctive", "previous_tags", "exemplarity",
    "selfclaim_lv", "selfclaim_cos", "selfclaim_jw",
))
FILTERS = frozenset(("all", "subsample_6m", "subsample_1m", "subsample_1w",
                     "subsample_q75", "subsample_q50", "subsample_q25", "subsample_top"))
MODEL_FIELDS = PREDICTORS | (FILTERS - {"all"})
KEYS = frozenset(("family", "outcome", "terms", "strata", "cluster", "filter"))


def _term(term):
    if not isinstance(term, str) or len(term) > 120:
        raise ValueError("Each term must be a short allowed expression")
    term = re.sub(r"\s+", "", term)
    if term in MODEL_FIELDS:
        return term, (term,)
    match = re.fullmatch(r"(?:I\(([a-z_]+)\^2\)|([a-z_]+)\^2)", term)
    if match and (variable := (match[1] or match[2])) in PREDICTORS:
        return f"I({variable}^2)", (variable,)
    match = re.fullmatch(r"poly\(([a-z_]+),2\)", term)
    if match and match[1] in PREDICTORS:
        return f"poly({match[1]},2)", (match[1],)
    match = re.fullmatch(r"([a-z_][a-z0-9_]*):([a-z_][a-z0-9_]*)", term)
    if match and match[1] in MODEL_FIELDS and match[2] in MODEL_FIELDS and match[1] != match[2]:
        variables = tuple(sorted((match[1], match[2])))
        return ":".join(variables), variables
    raise ValueError("Allowed terms: predictor, I(predictor^2), poly(predictor,2), predictor:predictor")


def term_variables(term):
    """Return only allowlisted variables referenced by one valid term."""
    return _term(term)[1]


def canonical_term(term):
    """Return the unique allowlisted spelling for an equivalent analysis term."""
    return _term(term)[0]


def validate_action(action):
    """Return a normalized action or raise ValueError; no data access or evaluation."""
    if not isinstance(action, dict) or set(action) != KEYS:
        raise ValueError("Action requires exactly family/outcome/terms/strata/cluster/filter")
    for key in KEYS - {"terms"}:
        if not isinstance(action[key], str):
            raise ValueError(f"{key} must be a string")
    if action["family"] not in ("linear", "logit", "conditional_logit"):
        raise ValueError("Unsupported family")
    if action["outcome"] != "tagged":
        raise ValueError("Only tagged is an allowed outcome")
    if action["strata"] not in ("none", "choice_set") or action["cluster"] not in ("none", "user_label"):
        raise ValueError("Unsupported strata or cluster")
    if action["filter"] not in FILTERS:
        raise ValueError("Unsupported filter")
    if action["family"] == "conditional_logit" and action["strata"] != "choice_set":
        raise ValueError("conditional_logit requires choice_set strata")
    if not isinstance(action["terms"], list) or not 1 <= len(action["terms"]) <= 12:
        raise ValueError("Choose 1 to 12 terms")
    # The explicitly requested group effect already absorbs its own identifier.
    terms = [_term(term)[0] for term in action["terms"]
             if not (action["strata"] == "choice_set" and isinstance(term, str)
                     and re.sub(r"\s+", "", term) == "choice_set")]
    terms = list(dict.fromkeys(terms))
    if not terms:
        raise ValueError("At least one explanatory term is required beyond the grouping identifier")
    return {**action, "terms": terms}


def render_r(action):
    """Return standalone R source; caller must run it inside reviewed isolation."""
    action = validate_action(action)
    variables = {"tagged"}
    for term in action["terms"]:
        variables.update(term_variables(term))
    if action["strata"] != "none":
        variables.add(action["strata"])
    if action["cluster"] != "none":
        variables.add(action["cluster"])
    needed = ", ".join(f'"{variable}"' for variable in sorted(variables))
    numeric = ", ".join(f'"{variable}"' for variable in sorted(variables & (MODEL_FIELDS | {"tagged"})))
    formula = "tagged ~ " + " + ".join(action["terms"])
    if action["strata"] != "none":
        formula += " + strata(choice_set)"
    subset = "d" if action["filter"] == "all" else f'd[!is.na(d${action["filter"]}) & d${action["filter"]} == TRUE, , drop=FALSE]'
    family = action["family"]
    fixed_effect = family != "conditional_logit" and action["strata"] != "none"
    if fixed_effect:
        formula = "tagged ~ " + " + ".join(action["terms"]) + " | choice_set"
        covariance = "~user_label" if action["cluster"] != "none" else '"iid"'
        if family == "linear":
            fitting = f'fixest::feols(formula, data=d, vcov={covariance}, nthreads=1, notes=FALSE)'
        else:
            fitting = f'fixest::feglm(formula, data=d, family=stats::binomial(), vcov={covariance}, nthreads=1, notes=FALSE)'
    elif family == "linear":
        fitting = 'stats::lm(formula, data=d, na.action=stats::na.fail)'
    elif family == "logit":
        fitting = 'stats::glm(formula, data=d, family=stats::binomial(), na.action=stats::na.fail, control=stats::glm.control(maxit=100))'
    else:
        cluster = ", cluster=user_label, robust=TRUE" if action["cluster"] != "none" else ", robust=FALSE"
        fitting = f'survival::clogit(formula, data=d, method="efron", na.action=stats::na.fail, control=survival::coxph.control(iter.max=50){cluster})'
    se_type = "cluster_robust" if action["cluster"] != "none" else "model_based"
    cluster_count = 'length(unique(d$user_label))' if action["cluster"] != "none" else 'NA_integer_'
    filter_column = "" if action["filter"] == "all" else f'if (!"{action["filter"]}" %in% names(d)) stop("Missing filter column")'
    return f'''# Generated from a validated bounded action; no arbitrary code input.
Sys.umask("0077")
Sys.setenv(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
suppressPackageStartupMessages(library(survival))
options(warn=1)
warnings_seen <- character()
metrics <- data.frame(n=NA_integer_, events=NA_integer_, clusters=NA_integer_,
                      loglik=NA_real_, SE_type="{se_type}", warnings="",
                      rows_input=NA_integer_, rows_after_filter=NA_integer_, complete_case_dropped=NA_integer_,
                      sample_membership_md5="")
write_metrics <- function() {{
  metrics$warnings <- paste(unique(warnings_seen), collapse=" | ")
  utils::write.csv(metrics, "/work/metrics.csv", row.names=FALSE, na="NA")
}}
tryCatch(withCallingHandlers({{
  d <- utils::read.csv("/data/styles.csv", stringsAsFactors=FALSE, check.names=FALSE,
                       na.strings=c("", "NA", "NaN"))
  metrics$rows_input <- nrow(d)
  needed <- c({needed})
  if (anyDuplicated(names(d)) || !all(needed %in% names(d))) stop("Invalid or missing input columns")
  {filter_column}
  d <- {subset}
  metrics$rows_after_filter <- nrow(d)
  numeric_columns <- c({numeric})
  if (!all(vapply(d[numeric_columns], function(x) is.numeric(x) || is.logical(x), logical(1)))) stop("Predictors/outcome must be numeric")
  for (column in numeric_columns) if (is.logical(d[[column]])) d[[column]] <- as.integer(d[[column]])
  usable <- stats::complete.cases(d[needed])
  for (column in numeric_columns) usable <- usable & is.finite(d[[column]])
  d <- d[usable, , drop=FALSE]
  metrics$complete_case_dropped <- metrics$rows_after_filter - nrow(d)
  metrics$n <- nrow(d)
  if (!nrow(d)) stop("No complete observations")
  if (!all(d$tagged %in% c(0, 1))) stop("tagged must contain only zero or one")
  metrics$events <- sum(d$tagged)
  metrics$clusters <- {cluster_count}
  if ("{se_type}" == "cluster_robust" && metrics$clusters < 2) stop("At least two clusters required")
  if (length(unique(d$tagged)) < 2) stop("Outcome has no variation")
  strata <- survival::strata
  formula <- stats::as.formula("{formula}")
  fit <- {fitting}
  sample_rows <- rownames(d)
  if (inherits(fit, "glm") && !isTRUE(fit$converged)) stop("GLM failed to converge")
  table <- if (inherits(fit, "fixest")) summary(fit)$coeftable else summary(fit)$coefficients
  if (inherits(fit, "fixest")) {{
    used <- d[fixest::obs(fit), , drop=FALSE]
    sample_rows <- rownames(used)
    metrics$n <- nrow(used)
    metrics$events <- sum(used$tagged)
    if ("{se_type}" == "cluster_robust") metrics$clusters <- length(unique(used$user_label))
  }} else if ("{family}" != "conditional_logit" && "{se_type}" == "cluster_robust") {{
    covariance <- sandwich::vcovCL(fit, cluster=d$user_label, type="HC1")
    reference_df <- if (inherits(fit, "glm")) Inf else metrics$clusters-1
    table <- lmtest::coeftest(fit, vcov.=covariance, df=reference_df)
  }}
  se_column <- if ("robust se" %in% colnames(table)) "robust se" else if ("se(coef)" %in% colnames(table)) "se(coef)" else "Std. Error"
  p_column <- if ("Pr(>|z|)" %in% colnames(table)) "Pr(>|z|)" else "Pr(>|t|)"
  estimates <- data.frame(term=rownames(table), coefficient=table[, 1],
                          se=table[, se_column], p_value=table[, p_column], row.names=NULL)
  # Simple quadratic slopes only: interactions and transformed bases need their own contrast.
  for (variable in c({', '.join(chr(34) + x + chr(34) for x in sorted(variables & PREDICTORS))})) {{
    linear <- variable
    quadratic <- paste0("I(", variable, "^2)")
    if (inherits(fit, "fixest") && !(quadratic %in% names(stats::coef(fit)))) quadratic <- paste0("I(", quadratic, ")")
    if (!(linear %in% names(stats::coef(fit))) || !(quadratic %in% names(stats::coef(fit)))) next
    if (any(grepl(":", rownames(table), fixed=TRUE) & grepl(paste0("(^|:)", variable, "(:|$)"), rownames(table)))) next
    covariance <- if (inherits(fit, "fixest")) stats::vcov(fit, vcov={covariance if fixed_effect else '"iid"'}) else if ("{family}" != "conditional_logit" && "{se_type}" == "cluster_robust") covariance else stats::vcov(fit)
    used <- if (inherits(fit, "fixest")) d[fixest::obs(fit), , drop=FALSE] else d
    limits <- range(used[[variable]], na.rm=TRUE)
    b <- stats::coef(fit)[c(linear, quadratic)]
    v <- covariance[c(linear, quadratic), c(linear, quadratic), drop=FALSE]
    if (any(!is.finite(c(limits, b, v)))) next
    prefix <- paste0("quadratic_", variable, "_")
    metrics[[paste0(prefix, "sample_min")]] <- limits[1]
    metrics[[paste0(prefix, "sample_max")]] <- limits[2]
    metrics[[paste0(prefix, "cov_linear_quadratic")]] <- v[1, 2]
    metrics[[paste0(prefix, "turning_point")]] <- if (b[2] != 0) -b[1] / (2*b[2]) else NA_real_
    for (i in seq_along(limits)) {{
      gradient <- c(1, 2*limits[i])
      slope <- sum(gradient*b)
      variance <- as.numeric(t(gradient) %*% v %*% gradient)
      if (!is.finite(variance) || variance < 0) next
      label <- if (i == 1) "min" else "max"
      metrics[[paste0(prefix, "slope_", label)]] <- slope
      metrics[[paste0(prefix, "slope_se_", label)]] <- sqrt(variance)
    }}
  }}
  if (anyNA(stats::coef(fit))) warnings_seen <- c(warnings_seen, "Non-estimable coefficients")
  metrics$loglik <- as.numeric(stats::logLik(fit))
  writeLines(sample_rows, "/work/sample-membership.txt")
  metrics$sample_membership_md5 <- unname(tools::md5sum("/work/sample-membership.txt"))
  utils::write.csv(estimates, "/work/estimates.csv", row.names=FALSE, na="NA")
  write_metrics()
}}, warning=function(w) {{
  warnings_seen <<- c(warnings_seen, conditionMessage(w))
  invokeRestart("muffleWarning")
}}), error=function(e) {{
  warnings_seen <<- c(warnings_seen, paste("ERROR:", conditionMessage(e)))
  write_metrics()
  stop(conditionMessage(e), call.=FALSE)
}})
'''
