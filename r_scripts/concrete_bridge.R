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
## .concrete_args — the single builder for the formatArguments analysis plan (the shared seam every run_*
## entry should use). Centralises: id/event_type/A coercion, the L1 -> CensoringTV routing (the collider-
## avoidance invariant, previously copy-pasted ~5x), and the horizon cap; plus the #223 fixes — a DENSE RMST
## TargetTime grid (a single target time cannot integrate the RMST integral ∫S dt) and the MinNuisance
## positivity bound (lower-bounds the propensity g so the TMLE clever covariate 1/g cannot blow up on
## overlap tails; the residual bias grew with n from exactly that). Params mirror ConcreteConfig.to_r_kwargs()
## in causal_bench/estimators/concrete_config.py. Returns list(args, horizon, dt).
## `rmst_grid=FALSE` gives a single TargetTime (for callers that pass their own grid / don't integrate RMST).
## ---------------------------------------------------------------------------
.concrete_args <- function(df, horizon, covars = c("W1", "W2", "W3", "W4"),
                           crossover_col = NULL, strata_cols = NULL,
                           cv_folds = 5, min_nuisance = NULL, rmst_grid_n = 12,
                           rmst_grid = TRUE, target_time = NULL, verbose = FALSE) {
  dt <- as.data.table(df)
  dt[, id := .I]; dt[, event_type := as.integer(event_type)]; dt[, A := as.integer(A)]
  ctv <- NULL
  if ("L1" %in% names(dt) && any(!is.na(dt[["L1"]]))) {
    obs_idx <- !is.na(dt[["L1"]])
    ctv <- data.table(id = dt$id[obs_idx], time = 0.5, L1 = dt[["L1"]][obs_idx])
    if (verbose) message(".concrete_args: L1 -> CensoringTV (not the outcome model)")
  }
  dt[, grep("^L[0-9]+$", names(dt), value = TRUE) := NULL]
  last_event <- max(dt[event_type == 1L, T_obs], na.rm = TRUE)
  if (horizon >= last_event) {
    horizon <- last_event * 0.999
    if (verbose) message(sprintf(".concrete_args: horizon capped to %.6f (last event)", horizon))
  }
  ## TargetTime: an explicit `target_time` (e.g. the simultaneous multi-horizon vector) wins; else a dense
  ## RMST grid; else the single (capped) horizon. `target_time` is still capped to the last event above.
  tt <- if (!is.null(target_time)) pmin(target_time, horizon)
        else if (isTRUE(rmst_grid) && rmst_grid_n >= 2)
          seq(horizon / rmst_grid_n, horizon, length.out = as.integer(rmst_grid_n)) else horizon
  fa <- list(
    DataTable = dt, EventTime = "T_obs", EventType = "event_type", Treatment = "A", ID = "id",
    Intervention = list(`1` = 1L, `0` = 0L), TargetTime = tt, TargetEvent = 1L,
    Covariates = covars, CVArg = list(V = as.integer(cv_folds)), CensoringTV = ctv,
    Crossover = crossover_col, Strata = strata_cols, Verbose = verbose)
  ## MinNuisance is the positivity bound — omitted (concrete's own default) unless explicitly set, so the
  ## structural fix stays positivity-neutral; the positivity fix supplies a value once verified (#223).
  if (!is.null(min_nuisance)) fa$MinNuisance <- as.numeric(min_nuisance)
  args <- tryCatch(do.call(concrete::formatArguments, fa),
                   error = function(e) stop(".concrete_args: formatArguments failed: ", conditionMessage(e)))
  list(args = args, horizon = horizon, dt = dt)
}


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
                                horizon      = 1.0,
                                covars       = c("W1", "W2", "W3", "W4"),
                                crossover_col = NULL,
                                strata_cols  = NULL,
                                cv_folds     = 5,
                                min_nuisance = NULL,
                                rmst_grid_n  = 12,
                                verbose      = FALSE) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)

  ## Build the analysis plan via the shared helper (L1->CensoringTV, horizon cap, dense RMST grid,
  ## MinNuisance positivity bound — see .concrete_args + #223).
  ca      <- .concrete_args(df, horizon, covars = covars, crossover_col = crossover_col,
                            strata_cols = strata_cols, cv_folds = cv_folds,
                            min_nuisance = min_nuisance, rmst_grid_n = rmst_grid_n, verbose = verbose)
  args    <- ca$args
  horizon <- ca$horizon
  dt      <- ca$dt

  ## Fit nuisance models + TMLE update
  est <- tryCatch(
    concrete::doConcrete(args),
    error = function(e) stop("concrete::doConcrete failed: ", conditionMessage(e))
  )

  ## Positivity / inverse-weight diagnostics (PR #28: getPositivityDx)
  pos_dx <- tryCatch(
    concrete::getPositivityDx(est, Verbose = verbose),
    error = function(e) NULL   # graceful fallback for older concrete versions
  )

  ## Extract RMST contrast — try getRMST() first (current API), fall back to
  ## targetRMST() for older installed versions.
  rmst <- tryCatch(
    ## Intervention = c(1L, 2L): the two intervention SLOTS (1-based) of Intervention=list('1','0'), i.e. the
    ## A=1 vs A=0 contrast. The old c(1L,0L) referenced a non-existent slot 0 — erroring on a grid, silently
    ## falling through to the LYL parse on a single time (the 0.221 "attenuation"). #223.
    concrete::getRMST(est, Horizon = horizon, Intervention = c(1L, 2L)),
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
  ## Parse result — extract the risk-difference contrast (treated − control).
  ##
  ## getRMST() returns per-arm rows for RMST and "Life Years Lost" (cause-
  ## specific CIF). There is no pre-computed contrast row. We prefer the
  ## "Life Years Lost" estimand (cause-specific CIF difference) as it is
  ## comparable to the Python estimators' pointwise risk difference.
  ##
  ## Fall-through order:
  ##   1. Pre-computed contrast row (Diff/RD/ATE in first column)
  ##   2. "Life Years Lost" per-arm rows → contrast by subtraction
  ##   3. "RMST" per-arm rows → contrast by subtraction
  ##   4. Legacy list layouts
  ## ---------------------------------------------------------------------------
  point <- NA_real_
  se    <- NA_real_

  .pt_col <- function(df) grep("Pt\\.Est|Pt Est|Estimate|Point|Est$",
                                names(df), value = TRUE)[1]
  .se_col <- function(df) grep("^SE$|^se$|Std\\.Err|StdErr",
                                names(df), value = TRUE)[1]
  .arm_col <- function(df) grep("Intervention|Arm|Treatment|arm",
                                 names(df), value = TRUE)[1]
  .est_col <- function(df) grep("Estimand|estimand",
                                 names(df), value = TRUE)[1]

  if (is.data.frame(rmst) || is.data.table(rmst)) {
    rmst_df <- as.data.frame(rmst)

    ## 1. Pre-computed contrast row
    diff_rows <- rmst_df[grepl("Diff|RD|ATE|diff|contrast",
                               rmst_df[[1]], ignore.case = TRUE), ]
    if (nrow(diff_rows) > 0) {
      pc <- .pt_col(diff_rows); sc <- .se_col(diff_rows)
      if (!is.na(pc)) point <- as.numeric(diff_rows[[pc]][1])
      if (!is.na(sc)) se    <- as.numeric(diff_rows[[sc]][1])
    }

    ## 2. Per-arm rows — compute contrast
    if (is.na(point)) {
      ac <- .arm_col(rmst_df); ec <- .est_col(rmst_df)
      pc <- .pt_col(rmst_df);  sc <- .se_col(rmst_df)

      if (!is.na(ac) && !is.na(pc)) {
        ## Prefer "Life Years Lost" (cause-specific CIF) over RMST
        for (estimand_pat in c("Life Years Lost", "RMST")) {
          if (!is.na(ec)) {
            sub <- rmst_df[grepl(estimand_pat, rmst_df[[ec]], ignore.case = TRUE), ]
          } else {
            sub <- rmst_df
          }
          ## Match arm labels: treated=1, control=0
          arms <- as.character(sub[[ac]])
          trt_row <- sub[grepl("A=1|=1|trt|treat|1$", arms, ignore.case = TRUE), ]
          ctl_row <- sub[grepl("A=0|=0|ctrl|control|0$", arms, ignore.case = TRUE), ]
          if (nrow(trt_row) > 0 && nrow(ctl_row) > 0) {
            est1 <- as.numeric(trt_row[[pc]][1])
            est0 <- as.numeric(ctl_row[[pc]][1])
            point <- est1 - est0
            if (!is.na(sc)) {
              se1 <- as.numeric(trt_row[[sc]][1])
              se0 <- as.numeric(ctl_row[[sc]][1])
              se  <- sqrt(se1^2 + se0^2)   # delta method (independent arms)
            }
            break
          }
        }
      }
    }

  } else if (is.list(rmst)) {
    ## 4. Legacy list layouts
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

  ## Additively parse the pre-computed RMST-Diff contrast row (Estimand == "RMST Diff"): treated − control,
  ## positive = survival BENEFIT, NO negation (unlike the LYL/risk-difference `ATE` above, which the Python
  ## side negates). Now reachable given the Intervention=c(1,2) + dense-grid fix. Consumers opt in via
  ## ConcreteRMSTEstimator(rmst_contrast=True) so the existing ATE/SE contract is untouched. #223 (Fix 1).
  rmst_point <- NA_real_; rmst_se <- NA_real_
  if (is.data.frame(rmst) || is.data.table(rmst)) {
    rdf2 <- as.data.frame(rmst)
    ec   <- .est_col(rdf2)
    if (!is.na(ec)) {
      dr <- rdf2[grepl("RMST Diff", rdf2[[ec]], ignore.case = TRUE), ]
      if (nrow(dr) > 0) {
        pc <- .pt_col(dr); sc <- .se_col(dr)
        if (!is.na(pc)) rmst_point <- as.numeric(dr[[pc]][1])
        if (!is.na(sc)) rmst_se    <- as.numeric(dr[[sc]][1])
      }
    }
  }

  converged <- !is.na(point) && is.finite(point) && is.finite(se)

  list(
    ATE          = point,
    SE           = se,
    RMST_ATE     = rmst_point,   # treated−control RMST diff (survival benefit; NO negation) — #223 Fix 1
    RMST_SE      = rmst_se,
    CI_lower     = point - 1.96 * se,
    CI_upper     = point + 1.96 * se,
    converged    = converged,
    positivity   = if (!is.null(pos_dx)) pos_dx$summary else NULL,
    raw          = rmst
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
##   df            same data.frame as run_concrete_bridge expects
##   horizon       target time
##   deltas        numeric vector of delta values in [0, 1] (default 0..0.20)
##   mechanism     "all" | "dropout" | "crossover" — which censoring pool to tip
##                 (PR #28; "crossover" requires a Crossover model)
##   crossover_col column name of per-subject switch times (NULL = ITT estimand)
##   covars        outcome covariate column names
##   verbose       passed through to concrete
##
## Returns a data.frame with columns:
##   mechanism, delta, estimate, se, ci_lower, ci_upper
##   attr "tipping_point" — concrete's own tipping-point value (list by event)
## ---------------------------------------------------------------------------
run_concrete_sensitivity <- function(df,
                                     horizon       = 1.0,
                                     deltas        = c(0, 0.05, 0.10, 0.15, 0.20),
                                     mechanism     = "all",
                                     crossover_col = NULL,
                                     strata_cols   = NULL,
                                     covars        = c("W1", "W2", "W3", "W4"),
                                     verbose       = FALSE) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)
  stopifnot(is.numeric(deltas), all(deltas >= 0), all(deltas <= 1))
  mechanism <- match.arg(mechanism, c("all", "dropout", "crossover"))

  ## Analysis plan via the shared helper (single horizon; senseCensoring does not integrate RMST).
  ca      <- .concrete_args(df, horizon, covars = covars, crossover_col = crossover_col,
                            strata_cols = strata_cols, rmst_grid = FALSE, verbose = verbose)
  args    <- ca$args
  horizon <- ca$horizon

  sens_raw <- tryCatch(
    concrete::senseCensoring(args, deltas = deltas, Estimand = "RD",
                             mechanism = mechanism),
    error = function(e) stop("concrete::senseCensoring failed: ", conditionMessage(e))
  )

  ## Stash concrete's own tipping-point attr before we reshape.
  tipping_pt <- attr(sens_raw, "tippingPoint")

  ## Normalize to a stable data.frame regardless of concrete version.
  ## PR #28 adds a leading "mechanism" column; preserve it when present.
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

  mech_col   <- .pick(c("^mechanism$", "^Mechanism$"))
  delta_col  <- .pick(c("^delta$", "^Delta$", "shift"))
  pt_col     <- .pick(c("Pt.Est", "Pt Est", "Estimate", "point", "coef"))
  se_col     <- .pick(c("^SE$", "StdErr", "Std.Err", "\\.SE$"))
  ci_lo_col  <- .pick(c("CI.Low", "CI Low", "lower", "\\.lo$", "lwr"))
  ci_hi_col  <- .pick(c("CI.Hi",  "CI Hi",  "upper", "\\.hi$", "upr"))

  out <- data.frame(
    mechanism = if (!is.na(mech_col))   as.character(dt_s[[mech_col]]) else mechanism,
    delta     = if (!is.na(delta_col))  as.numeric(dt_s[[delta_col]])  else deltas,
    estimate  = if (!is.na(pt_col))     as.numeric(dt_s[[pt_col]])     else NA_real_,
    se        = if (!is.na(se_col))     as.numeric(dt_s[[se_col]])     else NA_real_,
    ci_lower  = if (!is.na(ci_lo_col))  as.numeric(dt_s[[ci_lo_col]])  else NA_real_,
    ci_upper  = if (!is.na(ci_hi_col))  as.numeric(dt_s[[ci_hi_col]])  else NA_real_
  )

  ## Fill gaps: derive SE from CI or CI from SE
  missing_se <- is.na(out$se)
  missing_ci <- is.na(out$ci_lower)
  if (any(missing_se) && !any(missing_ci))
    out$se[missing_se] <- (out$ci_upper[missing_se] - out$ci_lower[missing_se]) / (2 * 1.96)
  if (any(missing_ci) && !any(missing_se)) {
    out$ci_lower[missing_ci] <- out$estimate[missing_ci] - 1.96 * out$se[missing_ci]
    out$ci_upper[missing_ci] <- out$estimate[missing_ci] + 1.96 * out$se[missing_ci]
  }

  attr(out, "tipping_point") <- tipping_pt
  out
}


## ---------------------------------------------------------------------------
## run_concrete_positivity_dx
##
## Wraps concrete::getPositivityDx() (PR #28): reports per-arm ESS, max
## weight, minimum observation probability, and truncation-bound share.
## Returns the $summary data.frame (one row per intervention) and $byTime
## (per-evaluation-time detail).
##
## Parameters
##   df        same data.frame as run_concrete_bridge expects
##   horizon   target time
##   covars    outcome covariate column names
##   crossover_col  per-subject switch-time column (NULL = ITT)
##   verbose   passed through to concrete
## ---------------------------------------------------------------------------
run_concrete_positivity_dx <- function(df,
                                       horizon       = 1.0,
                                       covars        = c("W1", "W2", "W3", "W4"),
                                       crossover_col = NULL,
                                       strata_cols   = NULL,
                                       verbose       = FALSE) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))

  ## Analysis plan via the shared helper (single horizon; positivity Dx is per-time on the doConcrete fit).
  ca   <- .concrete_args(df, horizon, covars = covars, crossover_col = crossover_col,
                         strata_cols = strata_cols, rmst_grid = FALSE, verbose = verbose)
  args <- ca$args

  est <- tryCatch(
    concrete::doConcrete(args),
    error = function(e) stop("concrete::doConcrete failed: ", conditionMessage(e))
  )

  concrete::getPositivityDx(est, Verbose = verbose)
}


## ---------------------------------------------------------------------------
## run_concrete_simultaneous
##
## Multi-horizon TMLE with simultaneous confidence bands via
## getSimultaneousFamily() (concrete PR #31 + #33).
##
## Calls doConcrete once with all requested horizons as TargetTime, then feeds
## the resulting ConcreteEst to each estimand function — getOutput(Estimand="RD"),
## getRMST(), getRMTIF(), getWinRatio() — each of which attaches per-subject
## influence functions (famEst / famIC) to its ConcreteOut object. Those objects
## are stacked by getSimultaneousFamily(), which builds the joint n×q IC matrix,
## estimates R̂ = cor(IC_matrix), draws the (1-α) quantile of max_j|Z_j| for
## Z ~ N(0,R̂) via 10 000 Gaussian-multiplier samples (MASS::mvrnorm), and
## returns SimCI Low/Hi for every row.
##
## The RD IC matrix is also extracted from attr(rd_out, "famIC") and returned
## separately for the Python-side Gaussian multiplier bootstrap used in Exp 12
## for non-concrete estimators.
##
## Parameters
##   df        data.frame from generate_data(); must have T_obs, event_type, A
##   horizons  numeric vector of target times (e.g. c(0.4, 0.7))
##   covars    outcome covariate column names
##   signif    significance level for simultaneous bands (default 0.05)
##   verbose   passed through to concrete
##
## Returns named list:
##   $results    data.frame — one row per scalar estimand in the family:
##                 estimand, horizon, point, se, ci_lo, ci_hi,
##                 sim_ci_lo, sim_ci_hi, sim_q
##               All rows have sim_ci_lo/hi from getSimultaneousFamily; RMTIF
##               and WR rows have horizon = max(horizons_used).
##   $ic_matrix  data.frame — n rows × q RD columns (RD_t<time>_ev<event>)
##               IC1 - IC0 from the doConcrete fit, for Python-side bootstrap.
##   $sim_q      numeric — simultaneous critical value (attr critValue from fam)
##   $tmle_diag  data.frame or NULL — getTmleDiagnostics output
##   $n          integer — sample size
##   $horizons   numeric — actual horizons used (capped to last event time)
##   $converged  logical
## ---------------------------------------------------------------------------
run_concrete_simultaneous <- function(df,
                                      horizons = c(0.4, 0.7),
                                      covars   = c("W1", "W2", "W3", "W4"),
                                      signif   = 0.05,
                                      verbose  = FALSE) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizons), length(horizons) >= 1)

  ## Analysis plan via the shared helper. All horizons as TargetTime (target_time override) so every
  ## estimand function sees the same doConcrete fit and subject IDs align for getSimultaneousFamily();
  ## the helper caps them to the last event, reproducing the old horizons_used.
  ca            <- .concrete_args(df, max(horizons), covars = covars,
                                  target_time = sort(horizons), verbose = verbose)
  args          <- ca$args
  dt            <- ca$dt
  horizons_used <- pmin(sort(horizons), ca$horizon)
  if (verbose && any(horizons != horizons_used))
    message("run_concrete_simultaneous: some horizons capped to ", ca$horizon)

  est <- tryCatch(
    concrete::doConcrete(args),
    error = function(e) stop("doConcrete failed: ", conditionMessage(e))
  )

  ## --- Per-estimand outputs (Simultaneous=FALSE so each attaches its own
  ## famIC without running its own bootstrap; getSimultaneousFamily does the
  ## joint bootstrap once across the whole family.) ---------------------------

  ## RD: getOutput with Estimand="RD" attaches famIC with IC1-IC0 per
  ## (time, event), keyed as "RD e<event> t<time>". These are the RD ICs
  ## that drive the simultaneous bands and the Python-side bootstrap.
  rd_out <- tryCatch(
    concrete::getOutput(est, Estimand = "RD", Intervention = c(1L, 2L),
                        GComp = FALSE, Simultaneous = FALSE, Signif = signif),
    error = function(e) { warning("getOutput RD failed: ", e$message); NULL }
  )

  ## RMST at the largest horizon (integrates over all TargetTime grid points).
  rmst_out <- tryCatch(
    concrete::getRMST(est, Horizon = max(horizons_used),
                      Intervention = c(1L, 2L), Signif = signif),
    error = function(e) { warning("getRMST failed: ", e$message); NULL }
  )

  ## RMT-IF (single-event competing-risks; reduces to RMST diff when K=1).
  ## TargetEvent=1L selects the primary cause; pass the priority vector for
  ## competing-risks when multiple events are targeted.
  rmtif_out <- tryCatch(
    concrete::getRMTIF(est, Horizon = max(horizons_used),
                       Intervention = c(1L, 2L),
                       TargetEvent  = attr(est, "TargetEvent")[1L],
                       Signif       = signif),
    error = function(e) { warning("getRMTIF failed: ", e$message); NULL }
  )

  ## Win ratio (optional: only meaningful when multiple event types are
  ## targeted; getWinRatio errors on single-event fits, so wrap gracefully).
  wr_out <- tryCatch(
    concrete::getWinRatio(est, Intervention = c(1L, 2L), Signif = signif),
    error = function(e) NULL     # silently absent for single-event DGPs
  )

  ## --- Joint simultaneous family -------------------------------------------
  ## Build argument list: only pass non-NULL outputs. Each object carries
  ## famEst / famIC (attached by .attachFamily inside the estimand functions)
  ## and must be produced from the same doConcrete() object so IDs align.
  fam_args <- list(Signif = signif)
  if (!is.null(rd_out))    fam_args[["RD"]]    <- rd_out
  if (!is.null(rmst_out))  fam_args[["RMST"]]  <- rmst_out
  if (!is.null(rmtif_out)) fam_args[["RMTIF"]] <- rmtif_out
  if (!is.null(wr_out))    fam_args[["WR"]]    <- wr_out

  fam <- tryCatch(
    do.call(concrete::getSimultaneousFamily, fam_args),
    error = function(e) {
      warning("getSimultaneousFamily failed: ", e$message, " — returning pointwise only")
      NULL
    }
  )

  sim_q <- if (!is.null(fam)) as.numeric(attr(fam, "critValue")) else NA_real_

  ## --- Parse getSimultaneousFamily output → results data.frame -------------
  results <- data.frame()
  if (!is.null(fam) && nrow(fam) > 0) {
    fdf <- as.data.frame(fam)
    cn  <- names(fdf)
    .c  <- function(...) {
      for (p in list(...)) { m <- grep(p, cn, value = TRUE, ignore.case = TRUE); if (length(m)) return(m[1]) }
      NA_character_
    }
    fam_col   <- .c("^family$")
    est_col   <- .c("^Estimand$")
    time_col  <- .c("^Time$")
    pt_col    <- .c("Pt.Est", "Pt Est")
    se_col    <- .c("^se$", "^SE$")
    cilo_col  <- .c("CI.Low", "CI Low")
    cihi_col  <- .c("CI.Hi",  "CI Hi")
    slo_col   <- .c("SimCI.Low", "SimCI Low")
    shi_col   <- .c("SimCI.Hi",  "SimCI Hi")

    ## Build a clean estimand label: "<family>_<Estimand>_t<Time>"
    ## For WR/RMTIF rows the Time may be the horizon; use it as-is.
    times <- if (!is.na(time_col)) as.numeric(fdf[[time_col]]) else rep(max(horizons_used), nrow(fdf))
    labels <- paste0(fdf[[fam_col]], "_", gsub(" ", "", fdf[[est_col]]), "_t", times)

    results <- data.frame(
      estimand  = labels,
      horizon   = times,
      point     = as.numeric(fdf[[pt_col]]),
      se        = as.numeric(fdf[[se_col]]),
      ci_lo     = if (!is.na(cilo_col)) as.numeric(fdf[[cilo_col]]) else NA_real_,
      ci_hi     = if (!is.na(cihi_col)) as.numeric(fdf[[cihi_col]]) else NA_real_,
      sim_ci_lo = if (!is.na(slo_col))  as.numeric(fdf[[slo_col]])  else NA_real_,
      sim_ci_hi = if (!is.na(shi_col))  as.numeric(fdf[[shi_col]])  else NA_real_,
      sim_q     = sim_q,
      stringsAsFactors = FALSE
    )
    rownames(results) <- NULL
  }

  ## --- IC matrix for the Python-side bootstrap (Exp 12 non-concrete) -------
  ## Extract the RD influence-function long table from rd_out's famIC attribute
  ## (ekey = "RD e<event> t<time>") and dcast to wide format with Python-
  ## friendly column names (RD_t<time>_ev<event>).
  ic_matrix <- tryCatch({
    fi <- attr(rd_out, "famIC")           # data.table: ekey, ID, ic
    if (is.null(fi) || nrow(fi) == 0L) stop("famIC is empty")
    ## ekey format: "RD e1 t0.4" → col name "RD_t0.4_ev1"
    fi <- data.table::copy(fi)
    fi[, col_nm := sub("RD e([0-9]+) t(.+)", "RD_t\\2_ev\\1", ekey)]
    ic_wide <- data.table::dcast(fi, ID ~ col_nm, value.var = "ic")
    as.data.frame(ic_wide[, !"ID"])
  }, error = function(e) {
    warning("IC matrix extraction from famIC failed: ", e$message)
    ## fall back to manual IC1 - IC0 extraction
    tryCatch({
      ic1 <- as.data.table(est[[1L]][["IC"]])[, .(ID, Time, Event, IC1 = IC)]
      ic0 <- as.data.table(est[[2L]][["IC"]])[, .(ID, Time, Event, IC0 = IC)]
      ic_rd <- merge(ic1, ic0, by = c("ID", "Time", "Event"))
      ic_rd[, IC_RD  := IC1 - IC0]
      ic_rd[, col_nm := paste0("RD_t", Time, "_ev", Event)]
      ic_wide <- dcast(ic_rd, ID ~ col_nm, value.var = "IC_RD")
      as.data.frame(ic_wide[, !"ID"])
    }, error = function(e2) data.frame())
  })

  ## --- TMLE convergence diagnostics (Exp 13) --------------------------------
  tmle_diag <- tryCatch(
    as.data.frame(concrete::getTmleDiagnostics(est)),
    error = function(e) NULL
  )

  list(
    results   = results,
    ic_matrix = ic_matrix,
    sim_q     = sim_q,
    tmle_diag = tmle_diag,
    n         = nrow(dt),
    horizons  = horizons_used,
    converged = nrow(results) > 0 && !all(is.na(results$point))
  )
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

## ---------------------------------------------------------------------------
## run_concrete_win_ratio
##
## Wraps concrete::targetWinRatio() (direct TMLE, PR #30) or
## concrete::getWinRatio() (plug-in) depending on `method`.
##
## targetWinRatio() fluctuates both arms' cause-specific hazards jointly to
## solve the win/loss EIF estimating equations directly — removes target-grid
## sensitivity and cuts WR bias ~5x vs. plug-in (PR #30 validation table).
##
## Parameters
##   df            same data.frame as run_concrete_bridge expects
##   horizon       target time
##   method        "direct" (targetWinRatio) | "plugin" (getWinRatio via doConcrete)
##   covars        outcome covariate column names
##   crossover_col per-subject switch-time column (NULL = ITT estimand)
##   strata_cols   randomization strata (NULL = iid SE)
##   verbose       passed through to concrete
##
## Returns named list:
##   $WR         numeric — win ratio (treated / control)
##   $SE         numeric — influence-function SE
##   $CI_lower   numeric
##   $CI_upper   numeric
##   $win_odds   numeric
##   $net_benefit numeric
##   $converged  logical
##   $raw        the full concrete result object
## ---------------------------------------------------------------------------
run_concrete_win_ratio <- function(df,
                                   horizon       = 1.0,
                                   method        = "direct",
                                   covars        = c("W1", "W2", "W3", "W4"),
                                   crossover_col = NULL,
                                   strata_cols   = NULL,
                                   verbose       = FALSE) {

  method <- match.arg(method, c("direct", "plugin"))
  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)

  ## Analysis plan via the shared helper (single horizon; the win ratio is evaluated at one target time).
  ca      <- .concrete_args(df, horizon, covars = covars, crossover_col = crossover_col,
                            strata_cols = strata_cols, rmst_grid = FALSE, verbose = verbose)
  args    <- ca$args
  horizon <- ca$horizon

  wr_raw <- tryCatch({
    if (method == "direct") {
      concrete::targetWinRatio(args)
    } else {
      est <- concrete::doConcrete(args)
      concrete::getWinRatio(est)
    }
  }, error = function(e) stop("concrete win ratio failed: ", conditionMessage(e)))

  ## Extract WR, SE, CI — handle data.frame or list layouts defensively
  wr_val      <- NA_real_
  se_val      <- NA_real_
  ci_lo       <- NA_real_
  ci_hi       <- NA_real_
  win_odds    <- NA_real_
  net_benefit <- NA_real_

  .pick_col <- function(df, patterns) {
    for (p in patterns) {
      m <- grep(p, names(df), value = TRUE, ignore.case = TRUE)
      if (length(m)) return(m[1])
    }
    NA_character_
  }

  if (is.data.frame(wr_raw) || is.data.table(wr_raw)) {
    df_wr  <- as.data.frame(wr_raw)
    wr_col <- .pick_col(df_wr, c("^WR$", "^Win.Ratio$", "WinRatio", "Estimate"))
    se_col <- .pick_col(df_wr, c("^SE$", "^se$", "Std\\.Err", "StdErr"))
    lo_col <- .pick_col(df_wr, c("CI.Low", "lower", "lwr", "\\.lo$"))
    hi_col <- .pick_col(df_wr, c("CI.Hi",  "upper", "upr", "\\.hi$"))
    wo_col <- .pick_col(df_wr, c("WinOdds", "Win.Odds", "win_odds"))
    nb_col <- .pick_col(df_wr, c("NetBenefit", "Net.Benefit", "net_benefit"))

    wr_rows <- df_wr[grepl("^WR$|WinRatio|Win Ratio", df_wr[[1]], ignore.case = TRUE), ]
    src <- if (nrow(wr_rows) > 0) wr_rows else df_wr[1, , drop = FALSE]

    if (!is.na(wr_col))  wr_val      <- as.numeric(src[[wr_col]][1])
    if (!is.na(se_col))  se_val      <- as.numeric(src[[se_col]][1])
    if (!is.na(lo_col))  ci_lo       <- as.numeric(src[[lo_col]][1])
    if (!is.na(hi_col))  ci_hi       <- as.numeric(src[[hi_col]][1])
    if (!is.na(wo_col))  win_odds    <- as.numeric(src[[wo_col]][1])
    if (!is.na(nb_col))  net_benefit <- as.numeric(src[[nb_col]][1])

  } else if (is.list(wr_raw)) {
    if (!is.null(wr_raw$WR))          wr_val      <- as.numeric(wr_raw$WR)
    if (!is.null(wr_raw$SE))          se_val      <- as.numeric(wr_raw$SE)
    if (!is.null(wr_raw$CI_lower))    ci_lo       <- as.numeric(wr_raw$CI_lower)
    if (!is.null(wr_raw$CI_upper))    ci_hi       <- as.numeric(wr_raw$CI_upper)
    if (!is.null(wr_raw$win_odds))    win_odds    <- as.numeric(wr_raw$win_odds)
    if (!is.null(wr_raw$net_benefit)) net_benefit <- as.numeric(wr_raw$net_benefit)
  }

  converged <- isTRUE(attr(wr_raw, "WRConverged")) ||
               (!is.na(wr_val) && is.finite(wr_val) && !is.na(se_val) && is.finite(se_val))

  ## Fill missing CI from SE using asymmetric log-scale CI
  if ((is.na(ci_lo) || is.na(ci_hi)) && !is.na(wr_val) && !is.na(se_val)) {
    ci_lo <- exp(log(wr_val) - 1.96 * se_val / wr_val)
    ci_hi <- exp(log(wr_val) + 1.96 * se_val / wr_val)
  }

  list(
    WR          = wr_val,
    SE          = se_val,
    CI_lower    = ci_lo,
    CI_upper    = ci_hi,
    win_odds    = win_odds,
    net_benefit = net_benefit,
    converged   = converged,
    raw         = wr_raw
  )
}


## ---------------------------------------------------------------------------
## run_clinical_rmtif
##
## Wraps concrete::clinicalRMTIF() (PR #33), which estimates the restricted
## mean time in favorable state (RMT-IF) via the multistate / illness-death
## engine using an analytic adjoint-value EIF — a different estimator from
## getRMTIF() (which post-processes a doConcrete fit). clinicalRMTIF takes
## raw illness/terminal event times directly, not a ConcreteEst object.
##
## Signature (concrete 1.1.1.9000+):
##   clinicalRMTIF(data, arm, illness.time, terminal.time, terminal.status,
##                 covariates, horizon, n.grid, n.folds, SL.library, Signif,
##                 id, censoring.tv, nBoot)
##
## causal_bench's competing_risks_base DGP has event_type in {0, 1, 2}:
##   1 = primary (non-fatal) event  -> mapped to "illness"
##   2 = competing (fatal) event    -> mapped to "terminal"
## A subject can experience at most one of {illness, terminal} (first wins),
## so illness.time is set for event_type==1 subjects only; everyone else's
## illness.time is NA (never ill). terminal.time/terminal.status follow
## death-priority: only event_type==2 sets terminal.status=1.
##
## For single-event DGPs (no event_type==2 ever observed) this degenerates
## to a standard right-censored death-only analysis (illness.time always NA).
##
## Parameters
##   df       data.frame with T_obs, event_type, A (event_type in {0,1,2})
##   horizon  target time
##   covars   covariate column names
##   signif   significance level
##   verbose  unused (clinicalRMTIF has no Verbose arg); kept for API symmetry
##
## Returns named list:
##   $point      numeric — RMT-IF treated-vs-control contrast
##   $se         numeric
##   $ci_lower   numeric
##   $ci_upper   numeric
##   $converged  logical
##   $raw        the full concrete::clinicalRMTIF() result object
## ---------------------------------------------------------------------------
run_clinical_rmtif <- function(df,
                               horizon        = 1.0,
                               covars         = c("W1", "W2", "W3", "W4"),
                               signif         = 0.05,
                               crossover_col  = NULL,
                               min_cens_surv  = 0.05,
                               verbose        = FALSE) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)

  dt <- as.data.table(df)
  dt[, id         := .I]
  dt[, event_type := as.integer(event_type)]
  dt[, A          := as.integer(A)]

  ## Death-priority illness/terminal mapping (see header comment).
  dt[, illness_time  := ifelse(event_type == 1L, T_obs, NA_real_)]
  dt[, terminal_time := T_obs]
  dt[, terminal_status := as.integer(event_type == 2L)]

  crossover_arg <- if (is.null(crossover_col) || length(crossover_col) == 0) NULL else
                     as.character(crossover_col)[1L]

  ## TODO(#36): uncomment both lines marked below once concrete PR #36 merges.
  result <- tryCatch(
    concrete::clinicalRMTIF(
      data             = as.data.frame(dt),
      arm              = "A",
      illness.time     = "illness_time",
      terminal.time    = "terminal_time",
      terminal.status  = "terminal_status",
      covariates       = covars,
      horizon          = horizon,
      ## crossover     = crossover_arg,            # TODO: uncomment on concrete PR #36 merge
      ## min.cens.surv = as.numeric(min_cens_surv), # TODO: uncomment on concrete PR #36 merge
      Signif           = signif,
      id               = "id"
    ),
    error = function(e) stop("concrete::clinicalRMTIF failed: ", conditionMessage(e))
  )

  .pick_col <- function(d, patterns) {
    for (p in patterns) {
      m <- grep(p, names(d), value = TRUE, ignore.case = TRUE)
      if (length(m)) return(m[1])
    }
    NA_character_
  }

  point <- se <- ci_lo <- ci_hi <- NA_real_

  if (is.data.frame(result) || is.data.table(result)) {
    rdf    <- as.data.frame(result)
    pt_col <- .pick_col(rdf, c("Pt.Est", "Pt Est", "^estimate$", "^RMT.IF$"))
    se_col <- .pick_col(rdf, c("^se$", "^SE$"))
    lo_col <- .pick_col(rdf, c("CI.Low", "CI Low", "lower"))
    hi_col <- .pick_col(rdf, c("CI.Hi", "CI Hi", "upper"))
    row    <- rdf[1, , drop = FALSE]
    if (!is.na(pt_col)) point <- as.numeric(row[[pt_col]][1])
    if (!is.na(se_col)) se    <- as.numeric(row[[se_col]][1])
    if (!is.na(lo_col)) ci_lo <- as.numeric(row[[lo_col]][1])
    if (!is.na(hi_col)) ci_hi <- as.numeric(row[[hi_col]][1])
  } else if (is.list(result)) {
    if (!is.null(result$estimate)) point <- as.numeric(result$estimate)
    if (!is.null(result$se))       se    <- as.numeric(result$se)
    if (!is.null(result$ci.lower)) ci_lo <- as.numeric(result$ci.lower)
    if (!is.null(result$ci.upper)) ci_hi <- as.numeric(result$ci.upper)
  }

  if ((is.na(ci_lo) || is.na(ci_hi)) && !is.na(point) && !is.na(se)) {
    z     <- stats::qnorm(1 - signif / 2)
    ci_lo <- point - z * se
    ci_hi <- point + z * se
  }

  list(
    point     = point,
    se        = se,
    ci_lower  = ci_lo,
    ci_upper  = ci_hi,
    converged = !is.na(point) && !is.na(se) && is.finite(point) && is.finite(se),
    raw       = result
  )
}

## ---------------------------------------------------------------------------
## run_clinical_psnb — bridge to concrete::clinicalPSNB() (PR #34).
##
## clinicalPSNB replaces the implicit reach weights in the standard hierarchical
## win ratio with a user-supplied charter vector α, producing the
## priority-standardized net benefit (PSNB = Σ_k α_k Δ_k) and win ratio
## (PSWR = Σ_k α_k w_k / Σ_k α_k ℓ_k) together with IF-based CIs.
##
## Uses the same illness-death / multistate mapping as run_clinical_rmtif:
##   event_type==1 -> illness; event_type==2 -> terminal/death-priority.
##
## Output format (concrete PR #34): clinicalPSNB() returns a data.table
## (class ConcreteOut) with one row per estimand, keyed by the Estimand
## column: "PSNB", "PSWR", "Reach[D]", "NetBenefit[D]", etc. Columns are
## "Pt Est", "se", "CI Low", "CI Hi" (with spaces). PSWR CIs are log-scale
## (CI Low = pswr*exp(-z*slwr), CI Hi = pswr*exp(z*slwr)), already correct.
##
## Parameters
##   df       data.frame with T_obs, event_type, A (event_type in {0,1,2})
##   horizon  target time τ
##   charter  numeric vector of per-tier weights summing to 1 (default: equal)
##   covars   covariate column names
##   signif   significance level
##
## Returns named list:
##   $psnb          numeric — priority-standardized net benefit
##   $pswr          numeric — priority-standardized win ratio
##   $se_psnb       numeric
##   $se_pswr       numeric (delta-method SE on original PSWR scale)
##   $ci_lower_psnb numeric
##   $ci_upper_psnb numeric
##   $ci_lower_pswr numeric (log-scale CI, already exponentiated)
##   $ci_upper_pswr numeric
##   $converged     logical
##   $raw           the full concrete::clinicalPSNB() result object
##   $tier_components  data.frame of per-tier Reach and NetBenefit rows
## ---------------------------------------------------------------------------
run_clinical_psnb <- function(df,
                               horizon       = 1.0,
                               charter       = NULL,
                               covars        = c("W1", "W2", "W3", "W4"),
                               signif        = 0.05,
                               min_cens_surv = 0.05) {

  stopifnot(is.data.frame(df))
  stopifnot(all(c("T_obs", "event_type", "A") %in% names(df)))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)

  dt <- as.data.table(df)
  dt[, id          := .I]
  dt[, event_type  := as.integer(event_type)]
  dt[, A           := as.integer(A)]
  dt[, illness_time  := ifelse(event_type == 1L, T_obs, NA_real_)]
  dt[, terminal_time := T_obs]
  dt[, terminal_status := as.integer(event_type == 2L)]

  ## Default charter: equal weight across 2 tiers (death, illness).
  ## clinicalPSNB rescales internally but requires non-negative, non-zero.
  n_tiers <- 2L
  if (is.null(charter)) {
    charter <- rep(1.0 / n_tiers, n_tiers)
  } else {
    charter <- as.numeric(charter)
    if (abs(sum(charter) - 1.0) > 1e-9)
      stop("charter must sum to 1 (got ", sum(charter), ")")
  }

  ## TODO(#36): uncomment the line marked below once concrete PR #36 merges.
  result <- tryCatch(
    concrete::clinicalPSNB(
      data             = as.data.frame(dt),
      arm              = "A",
      illness.time     = "illness_time",
      terminal.time    = "terminal_time",
      terminal.status  = "terminal_status",
      covariates       = covars,
      horizon          = horizon,
      charter          = charter,
      ## min.cens.surv = as.numeric(min_cens_surv), # TODO: uncomment on concrete PR #36 merge
      Signif           = signif,
      id               = "id"
    ),
    error = function(e) stop("concrete::clinicalPSNB failed: ", conditionMessage(e))
  )

  ## clinicalPSNB returns a data.table (ConcreteOut) with one row per estimand.
  ## Columns: Estimand, "Pt Est", se, "CI Low", "CI Hi", pValue.
  ## PSWR CIs are already log-scale (CI Low = pswr*exp(-z*slwr)); no fallback needed.
  psnb <- pswr <- se_psnb <- se_pswr <- NA_real_
  ci_lo_psnb <- ci_hi_psnb <- ci_lo_pswr <- ci_hi_pswr <- NA_real_
  tier_components <- NULL

  rdf <- as.data.frame(result)
  .row <- function(est) rdf[rdf$Estimand == est, , drop = FALSE]
  .val <- function(row, col) {
    v <- row[[col]]
    if (length(v) == 0 || is.null(v)) NA_real_ else as.numeric(v[1])
  }

  nb_row <- .row("PSNB")
  wr_row <- .row("PSWR")

  if (nrow(nb_row) > 0) {
    psnb       <- .val(nb_row, "Pt Est")
    se_psnb    <- .val(nb_row, "se")
    ci_lo_psnb <- .val(nb_row, "CI Low")
    ci_hi_psnb <- .val(nb_row, "CI Hi")
  }
  if (nrow(wr_row) > 0) {
    pswr       <- .val(wr_row, "Pt Est")
    se_pswr    <- .val(wr_row, "se")
    ci_lo_pswr <- .val(wr_row, "CI Low")   # log-scale CI — already correct
    ci_hi_pswr <- .val(wr_row, "CI Hi")
  }

  ## Fallback normal-theory CI for PSNB only (PSWR CI is log-scale from concrete).
  if ((is.na(ci_lo_psnb) || is.na(ci_hi_psnb)) && !is.na(psnb) && !is.na(se_psnb)) {
    z          <- stats::qnorm(1 - signif / 2)
    ci_lo_psnb <- psnb - z * se_psnb
    ci_hi_psnb <- psnb + z * se_psnb
  }

  ## Tier-level diagnostics: rows whose Estimand starts with Reach or NetBenefit.
  tier_mask <- grepl("^Reach\\[|^NetBenefit\\[", rdf$Estimand)
  if (any(tier_mask)) tier_components <- rdf[tier_mask, , drop = FALSE]

  list(
    psnb          = psnb,
    pswr          = pswr,
    se_psnb       = se_psnb,
    se_pswr       = se_pswr,
    ci_lower_psnb = ci_lo_psnb,
    ci_upper_psnb = ci_hi_psnb,
    ci_lower_pswr = ci_lo_pswr,
    ci_upper_pswr = ci_hi_pswr,
    converged     = !is.na(psnb) && !is.na(pswr) && is.finite(psnb) && is.finite(pswr),
    raw           = result,
    tier_components = tier_components
  )
}


## ---------------------------------------------------------------------------
## run_concrete_pro_win_ratio   (concrete PR #35 + PR #36 — pro= tiers, GPC rewrite)
##
## Parameters
##   df               data.frame with survival + PRO marker columns
##   horizon          numeric scalar
##   illness_time     character vector of non-terminal event time columns
##                    (NULL for death-only hierarchy)
##   terminal_time    character scalar, terminal event time column
##   terminal_status  character scalar, 0/1 death indicator
##   covariates       character vector of covariate column names
##   pro_specs        R list of PRO spec lists, each with:
##                      marker, landmark (must equal horizon post-#36),
##                      margin, direction, type
##   crossover_col    character scalar or NULL — per-subject treatment-switch
##                    time column for the hypothetical no-switching estimand
##                    (concrete PR #36: IPCW = 1/(S_dropout * S_crossover)).
##                    Accepted here; wired into clinicalWinRatio() below once
##                    PR #36 merges (see TODO).
##
## Returns named list:
##   $WR         numeric — win ratio (treated / control)
##   $SE         numeric — delta-method SE on log scale, back-transformed
##   $CI_lower   numeric
##   $CI_upper   numeric
##   $n_tiers    integer — total tiers (hard-event + PRO)
##   $converged  logical
##   $raw        full ConcreteOut object
## ---------------------------------------------------------------------------
run_concrete_pro_win_ratio <- function(df,
                                       horizon,
                                       illness_time    = NULL,
                                       terminal_time   = "T_obs",
                                       terminal_status = "Delta",
                                       covariates      = c("W1", "W2", "W3", "W4"),
                                       pro_specs       = NULL,
                                       crossover_col   = NULL,
                                       min_cens_surv   = 0.05) {
  stopifnot(is.data.frame(df))
  stopifnot(is.numeric(horizon), length(horizon) == 1, horizon > 0)

  df[[terminal_status]] <- as.integer(df[[terminal_status]])
  df[["arm"]]           <- as.integer(df[["A"]])

  ## Resolve illness times: NULL → empty (death-only hierarchy)
  ill_cols <- if (is.null(illness_time) || length(illness_time) == 0) character(0) else
                as.character(illness_time)

  ## Unwrap single-element StrVectors from rpy2
  term_col      <- as.character(terminal_time)[1L]
  status_col    <- as.character(terminal_status)[1L]
  crossover_arg <- if (is.null(crossover_col) || length(crossover_col) == 0) NULL else
                     as.character(crossover_col)[1L]

  ## TODO(#36): uncomment both lines marked below once concrete PR #36 merges.
  ## crossover_arg and min_cens_surv are accepted above and wired through from
  ## Python; the R call below is the only remaining wiring needed.
  result <- clinicalWinRatio(
    data             = df,
    arm              = "arm",
    illness.time     = ill_cols,
    terminal.time    = term_col,
    terminal.status  = status_col,
    covariates       = as.character(covariates),
    horizon          = as.numeric(horizon),
    ## crossover     = crossover_arg,            # TODO: uncomment on concrete PR #36 merge
    ## min.cens.surv = as.numeric(min_cens_surv), # TODO: uncomment on concrete PR #36 merge
    pro              = pro_specs
  )

  rdf  <- as.data.frame(result)
  .val <- function(df, col) {
    if (!col %in% names(df)) return(NA_real_)
    v <- df[[col]][1L]
    if (is.null(v) || length(v) == 0) NA_real_ else as.numeric(v)
  }

  wr_row <- rdf[grepl("^WinRatio", rdf$Estimand, ignore.case = TRUE), , drop = FALSE]
  if (nrow(wr_row) == 0) wr_row <- rdf[1L, , drop = FALSE]

  wr   <- .val(wr_row, "Estimate")
  se   <- .val(wr_row, "se")
  lo   <- .val(wr_row, "CI Low")
  hi   <- .val(wr_row, "CI Hi")

  list(
    WR        = wr,
    SE        = se,
    CI_lower  = lo,
    CI_upper  = hi,
    n_tiers   = as.integer(attr(result, "Tiers")),
    converged = !is.na(wr) && is.finite(wr),
    raw       = result
  )
}
