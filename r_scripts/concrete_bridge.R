## concrete_bridge.R
##
## Two usage modes:
##
##   1. Called from Python via rpy2 (causal_bench/estimators/concrete_rmst.py)
##      rpy2 sources this file and calls run_concrete_bridge(r_df, horizon)
##
##   2. Sourced directly in RStudio via reticulate — McCoy can call
##      generate_data() from Python and pipe the result here:
##
##        library(reticulate)
##        use_virtualenv("path/to/causal_bench/.venv")
##        cb <- import("causal_bench.dgp.survival")
##        cfg <- import("causal_bench.dgp.config")$DGPConfig(n=500L, collider_strength=0.5)
##        py_df <- cb$generate_data(cfg)
##        df <- as.data.frame(py_df)
##        df$event_type <- as.integer(df$Delta)
##        source("r_scripts/concrete_bridge.R")
##        result <- run_concrete_bridge(df, horizon = 1.0)

suppressPackageStartupMessages({
  library(concrete)
  library(data.table)
})


## ---------------------------------------------------------------------------
## run_concrete_bridge
##
## Parameters
##   df        data.frame with columns: T_obs, event_type (int), A (int),
##             W1, W2, W3, W4, [L1 optional]
##   horizon   numeric scalar — target time for RMST
##   covars    character vector of covariate column names (default W1-W4)
##
## Returns named list:
##   $ATE           numeric — RMST difference (treated - control)
##   $SE            numeric — standard error
##   $CI_lower      numeric
##   $CI_upper      numeric
##   $converged     logical
##   $raw           the full concrete result object (for debugging)
## ---------------------------------------------------------------------------
run_concrete_bridge <- function(df,
                                horizon    = 1.0,
                                covars     = c("W1", "W2", "W3", "W4"),
                                verbose    = FALSE) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)

  ## concrete requires a data.table
  dt <- as.data.table(df)

  ## event_type: 0 = censored, 1 = event of interest, 2 = competing event
  ## For single-cause survival, event_type ∈ {0, 1}
  dt[, event_type := as.integer(event_type)]
  dt[, A          := as.integer(A)]

  ## Build covariate formula — include L1 only if present and not all-NA
  use_covars <- covars
  if ("L1" %in% names(dt) && !all(is.na(dt[["L1"]]))) {
    use_covars <- c(use_covars, "L1")
    if (verbose) message("concrete_bridge: including L1 in covariate set")
  }

  ## formatArguments — wraps data + analysis plan into a single object
  ## NOTE: API may evolve as McCoy adds post-randomization censoring predictors.
  ##       The CensoringCovariates argument (below) is the extension point.
  args <- tryCatch(
    concrete::formatArguments(
      DataTable          = dt,
      EventTime          = "T_obs",
      EventType          = "event_type",
      Treatment          = "A",
      Intervention       = list(`1` = 1L, `0` = 0L),
      TargetTime         = horizon,
      TargetEvent        = 1L,          # cause of interest
      Covariates         = use_covars,
      ## CensoringCovariates — placeholder for McCoy's time-varying extension.
      ## Once concrete >= 1.x exposes this, pass L1 here instead of Covariates.
      ## CensoringCovariates = if ("L1" %in% names(dt)) "L1" else NULL,
      CVFolds            = 5L,
      Verbose            = verbose
    ),
    error = function(e) stop("concrete::formatArguments failed: ", conditionMessage(e))
  )

  ## Fit nuisance models + TMLE update
  est <- tryCatch(
    concrete::doConcrete(args),
    error = function(e) stop("concrete::doConcrete failed: ", conditionMessage(e))
  )

  ## Extract RMST contrast
  rmst <- tryCatch(
    concrete::getOutput(est, Estimand = "RMST", Contrast = "Treatment"),
    error = function(e) {
      ## Older concrete versions used targetRMST()
      if (verbose) message("getOutput failed, trying targetRMST()")
      concrete::targetRMST(est)
    }
  )

  ## ---------------------------------------------------------------------------
  ## Parse result — concrete's output structure varies by version.
  ## We try a sequence of known layouts and warn if none match.
  ## ---------------------------------------------------------------------------
  point <- NA_real_
  se    <- NA_real_

  if (is.list(rmst)) {
    ## Layout A: list with $Estimate and $SE at top level
    if (!is.null(rmst$Estimate)) {
      point <- as.numeric(rmst$Estimate)
      se    <- as.numeric(rmst$SE)

    ## Layout B: list with $Results$ATE$Estimate
    } else if (!is.null(rmst$Results$ATE$Estimate)) {
      point <- as.numeric(rmst$Results$ATE$Estimate)
      se    <- as.numeric(rmst$Results$ATE$SE)

    ## Layout C: data.frame/data.table row
    } else if (is.data.frame(rmst) || is.data.table(rmst)) {
      point <- as.numeric(rmst[1, "Estimate"])
      se    <- as.numeric(rmst[1, "SE"])

    } else {
      warning("concrete_bridge: unrecognised result layout — returning NA")
    }
  }

  converged <- !is.na(point) && is.finite(point) && is.finite(se)

  list(
    ATE       = point,
    SE        = se,
    CI_lower  = point - 1.96 * se,
    CI_upper  = point + 1.96 * se,
    converged = converged,
    raw       = rmst          # keep for debugging layout changes
  )
}


## ---------------------------------------------------------------------------
## Minimal smoke test — run when this file is sourced directly
## (not when loaded by rpy2, which sets CONCRETE_BRIDGE_SOURCED)
## ---------------------------------------------------------------------------
if (!exists("CONCRETE_BRIDGE_SOURCED")) {
  if (interactive() && requireNamespace("concrete", quietly = TRUE)) {
    message("concrete_bridge.R loaded. Call run_concrete_bridge(df, horizon).")
  }
}
