## flexsurv_bridge.R
##
## Royston-Parmar flexible-parametric survival (flexsurv::flexsurvspline) as a
## conditional-survival NUISANCE for the single-arm pooled-Q subgroup RMST
## estimator (issue #188, Route A). Called from Python via rpy2
## (causal_bench/estimators/rp_spline_nuisance.py): rpy2 sources this file and
## calls run_flexsurv_survival(df, covars, subgroup_col, t_grid).
##
## This returns predicted S(t_k | W_i, S_i) on the estimator's time grid; the
## Python side then debiases it with our own per-subgroup one-step TMLE. RP is a
## nuisance here, NOT a plug-in final estimator (no CI is read off flexsurv).

suppressPackageStartupMessages({
  library(flexsurv)
  library(survival)
})

## ---------------------------------------------------------------------------
## run_flexsurv_survival
##
## Parameters
##   df           data.frame with T_obs, Delta (1=event, 0=censored),
##                the covariate columns in `covars`, and `subgroup_col`.
##   covars       character vector of baseline covariate column names (e.g. W1-W4).
##   subgroup_col name of the pre-specified subgroup column (a group-varying spline
##                time-effect via anc=list(gamma1=~S) gives a non-PH per-subgroup shape).
##   t_grid       numeric vector of times at which to predict survival.
##   k            number of internal spline knots (default 2 -> a flexible hazard).
##
## Returns an n x length(t_grid) matrix S[i, k] = S(t_grid[k] | W_i, S_i), or a
## 1x1 matrix of NA on failure (the Python side treats NA as "unavailable" and
## falls back to the logistic-hazard nuisance).
## ---------------------------------------------------------------------------
run_flexsurv_survival <- function(df, covars, subgroup_col, t_grid, k = 2) {
  out <- tryCatch({
    df <- as.data.frame(df)
    df[[subgroup_col]] <- factor(df[[subgroup_col]])
    # Only vary the spline time-effect by subgroup when it has >1 level.
    rhs <- paste(c(covars, subgroup_col), collapse = " + ")
    anc <- if (nlevels(df[[subgroup_col]]) > 1)
      stats::setNames(list(stats::as.formula(paste0("~", subgroup_col))), "gamma1")
    else NULL
    form <- stats::as.formula(paste0("Surv(T_obs, Delta) ~ ", rhs))
    fit <- flexsurv::flexsurvspline(form, data = df, k = k, scale = "hazard",
                                    anc = anc)

    # summary() returns one data.frame per newdata row (columns: time, est).
    nd <- df[, c(covars, subgroup_col), drop = FALSE]
    s <- summary(fit, newdata = nd, t = as.numeric(t_grid),
                 type = "survival", ci = FALSE, tidy = FALSE)
    S <- t(vapply(s, function(d) d$est, numeric(length(t_grid))))  # n x K
    dimnames(S) <- NULL
    S
  }, error = function(e) matrix(NA_real_, 1, 1))
  out
}
