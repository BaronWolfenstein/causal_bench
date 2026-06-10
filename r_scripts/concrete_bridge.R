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
##             W1, W2, W3, W4, [L1 optional — goes into censoring model only]
##   horizon   numeric scalar — target time for RMST
##   covars    character vector of outcome covariate column names (default W1-W4)
##             L1 is intentionally excluded: it is a post-treatment mediator and
##             conditioning on it in the outcome model creates collider bias.
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

  ## concrete requires a data.table; add a row ID for CensoringTV matching
  dt <- as.data.table(df)
  dt[, id         := .I]
  dt[, event_type := as.integer(event_type)]
  dt[, A          := as.integer(A)]

  ## Build CensoringTV from L1 if present.
  ## L1 is a post-treatment time-varying covariate that drives both the event
  ## and informative censoring. It must enter ONLY the censoring model — passing
  ## it to the outcome model as a covariate creates collider bias (the same trap
  ## Exp 5 demonstrates with cox_l1).
  ctv <- NULL
  if ("L1" %in% names(dt) && any(!is.na(dt[["L1"]]))) {
    obs_idx <- !is.na(dt[["L1"]])
    ctv <- data.table(id   = dt$id[obs_idx],
                      time = 0.5,               # t_L1 matches DGPConfig.t_L1
                      L1   = dt[["L1"]][obs_idx])
    if (verbose) message("concrete_bridge: passing L1 to CensoringTV (not outcome model)")
  }
  ## Drop L1 from the main table — it's been extracted to CensoringTV and
  ## formatArguments rejects any column with NaN values in the DataTable.
  dt[, grep("^L[0-9]+$", names(dt), value = TRUE) := NULL]

  ## Cap TargetTime at the last observed event — concrete errors if the
  ## horizon falls after all individuals are censored.
  last_event <- max(dt[event_type == 1L, T_obs], na.rm = TRUE)
  if (horizon >= last_event) {
    horizon <- last_event * 0.999
    if (verbose) message(sprintf("concrete_bridge: horizon capped to %.6f (last event time)", horizon))
  }

  ## formatArguments — wraps data + analysis plan into a single object.
  ## CensoringTV conditions the IPCW on L1 (LOCF + change-from-baseline),
  ## correcting informative-censoring bias without touching the outcome hazards.
  args <- tryCatch(
    concrete::formatArguments(
      DataTable   = dt,
      EventTime   = "T_obs",
      EventType   = "event_type",
      Treatment   = "A",
      ID          = "id",
      Intervention = list(`1` = 1L, `0` = 0L),
      TargetTime  = horizon,
      TargetEvent = 1L,
      Covariates  = covars,
      CVArg       = list(V = 5L),
      CensoringTV = ctv,
      Verbose     = verbose
    ),
    error = function(e) stop("concrete::formatArguments failed: ", conditionMessage(e))
  )

  ## Fit nuisance models + TMLE update
  est <- tryCatch(
    concrete::doConcrete(args),
    error = function(e) stop("concrete::doConcrete failed: ", conditionMessage(e))
  )

  ## Extract RMST contrast — try getRMST() first (current API), fall back to
  ## targetRMST() for older installed versions.
  rmst <- tryCatch(
    concrete::getRMST(est, Horizon = horizon, Intervention = c(1L, 0L)),
    error = function(e) {
      if (verbose) message("getRMST failed, trying targetRMST()")
      tryCatch(
        concrete::targetRMST(est, Horizon = horizon),
        error = function(e2) {
          ## last resort: getOutput with RMST estimand
          concrete::getOutput(est, Estimand = "RMST", Contrast = "Treatment")
        }
      )
    }
  )

  ## ---------------------------------------------------------------------------
  ## Parse result — extract the RMST difference (treated − control).
  ## concrete's output structure varies across versions; try known layouts.
  ## ---------------------------------------------------------------------------
  point <- NA_real_
  se    <- NA_real_

  if (is.data.frame(rmst) || is.data.table(rmst)) {
    ## getRMST / targetRMST data.frame layout: look for the "RMST Diff" row
    diff_rows <- rmst[grepl("Diff|RD|ATE|diff", rmst[[1]], ignore.case = TRUE), ]
    if (nrow(diff_rows) == 0) diff_rows <- rmst  # fall back to first row

    pt_col <- grep("Pt.Est|Estimate|Point|Est$", names(diff_rows), value = TRUE)[1]
    se_col <- grep("^SE$|Std.Err|StdErr|se$", names(diff_rows), value = TRUE)[1]
    if (!is.na(pt_col)) point <- as.numeric(diff_rows[[pt_col]][1])
    if (!is.na(se_col)) se    <- as.numeric(diff_rows[[se_col]][1])

  } else if (is.list(rmst)) {
    ## Legacy list layouts
    if (!is.null(rmst$Estimate)) {
      point <- as.numeric(rmst$Estimate)
      se    <- as.numeric(rmst$SE)
    } else if (!is.null(rmst$Results$ATE$Estimate)) {
      point <- as.numeric(rmst$Results$ATE$Estimate)
      se    <- as.numeric(rmst$Results$ATE$SE)
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
## run_concrete_sensitivity
##
## Wraps concrete::senseCensoring(): a 1-D delta-shift MAR sensitivity
## analysis. At each delta, a fraction delta of the censored patients are
## assumed to be counterfactual events, and the doubly-robust TMLE
## risk-difference is re-estimated under that assumption.
##
## Because the same CensoringTV / formatArguments call is used here as in
## run_concrete_bridge(), L1 (when present) already conditions the IPCW
## before the delta-shift is applied — the baseline is the L1-corrected
## estimate, not the naive IPCW estimate.
##
## Parameters
##   df        same data.frame as run_concrete_bridge expects
##   horizon   target time
##   deltas    numeric vector of delta values in [0, 1] (default 0..0.20)
##   covars    outcome covariate column names (L1 is excluded; goes to CensoringTV)
##   verbose   passed through to concrete
##
## Returns a data.frame with columns:
##   delta, estimate, se, ci_lower, ci_upper
## ---------------------------------------------------------------------------
run_concrete_sensitivity <- function(df,
                                     horizon  = 1.0,
                                     deltas   = c(0, 0.05, 0.10, 0.15, 0.20),
                                     covars   = c("W1", "W2", "W3", "W4"),
                                     verbose  = FALSE) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)
  stopifnot(is.numeric(deltas), all(deltas >= 0), all(deltas <= 1))

  ## Build data.table + id + CensoringTV (identical logic to run_concrete_bridge)
  dt <- as.data.table(df)
  dt[, id         := .I]
  dt[, event_type := as.integer(event_type)]
  dt[, A          := as.integer(A)]

  ctv <- NULL
  if ("L1" %in% names(dt) && any(!is.na(dt[["L1"]]))) {
    obs_idx <- !is.na(dt[["L1"]])
    ctv <- data.table(id   = dt$id[obs_idx],
                      time = 0.5,
                      L1   = dt[["L1"]][obs_idx])
  }
  dt[, grep("^L[0-9]+$", names(dt), value = TRUE) := NULL]

  last_event <- max(dt[event_type == 1L, T_obs], na.rm = TRUE)
  if (horizon >= last_event) horizon <- last_event * 0.999

  args <- tryCatch(
    concrete::formatArguments(
      DataTable   = dt,
      EventTime   = "T_obs",
      EventType   = "event_type",
      Treatment   = "A",
      ID          = "id",
      Intervention = list(`1` = 1L, `0` = 0L),
      TargetTime  = horizon,
      TargetEvent = 1L,
      Covariates  = covars,
      CVArg       = list(V = 5L),
      CensoringTV = ctv,
      Verbose     = verbose
    ),
    error = function(e) stop("concrete::formatArguments failed: ", conditionMessage(e))
  )

  sens_raw <- tryCatch(
    concrete::senseCensoring(args, deltas = deltas, Estimand = "RD"),
    error = function(e) stop("concrete::senseCensoring failed: ", conditionMessage(e))
  )

  ## Normalize to a stable data.frame regardless of concrete version.
  ## Try column-name patterns for delta, point estimate, SE, CI bounds.
  if (!(is.data.frame(sens_raw) || is.data.table(sens_raw))) {
    stop("senseCensoring returned unrecognised format (not a data.frame)")
  }
  dt_s <- as.data.frame(sens_raw)
  cn   <- names(dt_s)

  .pick <- function(patterns) {
    for (p in patterns) {
      m <- grep(p, cn, value = TRUE, ignore.case = TRUE)
      if (length(m)) return(m[1])
    }
    NA_character_
  }

  delta_col  <- .pick(c("delta", "Delta", "shift"))
  pt_col     <- .pick(c("Pt.Est", "Pt Est", "Estimate", "point", "coef"))
  se_col     <- .pick(c("^SE$", "StdErr", "Std.Err", "\\.SE$"))
  ci_lo_col  <- .pick(c("CI.Low", "CI Low", "lower", "\\.lo$", "lwr"))
  ci_hi_col  <- .pick(c("CI.Hi",  "CI Hi",  "upper", "\\.hi$", "upr"))

  out <- data.frame(
    delta    = if (!is.na(delta_col))  as.numeric(dt_s[[delta_col]])  else deltas,
    estimate = if (!is.na(pt_col))     as.numeric(dt_s[[pt_col]])     else NA_real_,
    se       = if (!is.na(se_col))     as.numeric(dt_s[[se_col]])     else NA_real_,
    ci_lower = if (!is.na(ci_lo_col))  as.numeric(dt_s[[ci_lo_col]])  else NA_real_,
    ci_upper = if (!is.na(ci_hi_col))  as.numeric(dt_s[[ci_hi_col]])  else NA_real_
  )

  ## Fill gaps: derive SE from CI or CI from SE
  missing_se  <- is.na(out$se)
  missing_ci  <- is.na(out$ci_lower)
  if (any(missing_se)  && !any(missing_ci))
    out$se[missing_se] <- (out$ci_upper[missing_se] - out$ci_lower[missing_se]) / (2 * 1.96)
  if (any(missing_ci)  && !any(missing_se)) {
    out$ci_lower[missing_ci] <- out$estimate[missing_ci] - 1.96 * out$se[missing_ci]
    out$ci_upper[missing_ci] <- out$estimate[missing_ci] + 1.96 * out$se[missing_ci]
  }

  out
}


## ---------------------------------------------------------------------------
## Minimal smoke test — run when this file is sourced directly
## (not when loaded by rpy2, which sets CONCRETE_BRIDGE_SOURCED)
## ---------------------------------------------------------------------------
if (!exists("CONCRETE_BRIDGE_SOURCED")) {
  if (interactive() && requireNamespace("concrete", quietly = TRUE)) {
    message("concrete_bridge.R loaded. Call run_concrete_bridge(df, horizon) or run_concrete_sensitivity(df, horizon, deltas).")
  }
}
