## cobalt_bridge.R
##
## Regulator-familiar covariate-balance cross-check via the `cobalt` R package
## (bal.tab / love.plot). This is a PACKAGE-TIME cross-check, NOT the primary
## balance diagnostic: cobalt produces the standard bal.tab reviewers expect,
## but it does NOT produce the region-resolved SMD / deep-R ESS map that is the
## load-bearing diagnostic here (see experiments/exp29_balance_diagnostics.py).
## Use cobalt for familiarity/credibility; use exp29 for the region-R view.
##
## Two usage modes:
##   1. Called from Python via rpy2 (causal_bench/diagnostics/cobalt.py)
##      rpy2 sources this file and calls run_cobalt_baltab(r_df, weights, covs).
##   2. Sourced directly in R:
##        source("r_scripts/cobalt_bridge.R")
##        run_cobalt_baltab(df, w, c("X1","X2","X3","X4","X5"))
##
## Contract: `df` has the covariate columns + a `treat` column (1 = Target
## Group, 0 = Baseline Cohort); `weights` is a numeric vector aligned to `df`
## rows (1.0 for Target rows, the propensity odds-weight for Baseline rows).

run_cobalt_baltab <- function(df, weights, covs) {
  if (!requireNamespace("cobalt", quietly = TRUE)) {
    stop("cobalt is not installed; install.packages('cobalt')")
  }
  treat <- as.integer(df[["treat"]])
  X <- df[, covs, drop = FALSE]
  bt <- cobalt::bal.tab(
    X, treat = treat, weights = as.numeric(weights),
    method = "weighting", s.d.denom = "pooled",
    stats = c("mean.diffs", "variance.ratios"), un = TRUE
  )
  # Return the per-covariate balance table as a plain data.frame for rpy2.
  bal <- as.data.frame(bt$Balance)
  bal$covariate <- rownames(bal)
  rownames(bal) <- NULL
  bal
}
