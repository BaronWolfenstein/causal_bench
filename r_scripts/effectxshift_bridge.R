## effectxshift_bridge.R
##
## rpy2 bridge to McCoy's EffectXShift R package.
## Sourced from causal_bench/estimators/effectxshift.py via rpy2.
##
## Implements post-selection inference for the baseline subgroup with the
## largest ATE relative to its complement, using CV-TMLE / AIPW on held-out
## folds to give repeated-sampling guarantees after subgroup selection.
##
## run_effectxshift_bridge(df, outcome_col, treatment_col, covariate_cols,
##                         n_folds, max_depth)
##
## Returns a named list:
##   $v_ate         numeric — ATE in the selected high-benefit subgroup V
##   $vc_ate        numeric — ATE in the complement V^c
##   $contrast      numeric — v_ate - vc_ate (the post-selection estimand)
##   $v_se          numeric — SE of v_ate
##   $vc_se         numeric — SE of vc_ate
##   $contrast_se   numeric — SE of contrast
##   $rule          character — the selected subgroup rule string
##   $converged     logical

suppressPackageStartupMessages({
  library(EffectXshift)
})

run_effectxshift_bridge <- function(df,
                                    outcome_col    = "Y",
                                    treatment_col  = "A",
                                    covariate_cols = c("W1", "W2", "W3", "W4"),
                                    n_folds        = 5L,
                                    max_depth      = 2L) {
  stopifnot(is.data.frame(df))

  outcome_col   <- as.character(outcome_col[1])
  treatment_col <- as.character(treatment_col[1])

  stopifnot(outcome_col   %in% names(df))
  stopifnot(treatment_col %in% names(df))
  stopifnot(all(covariate_cols %in% names(df)))

  Y <- as.numeric(df[[outcome_col]])
  A <- as.integer(df[[treatment_col]])
  W <- df[, covariate_cols, drop = FALSE]

  result <- tryCatch(
    run_rct_workflow(
      Y         = Y,
      A         = A,
      W         = W,
      n_folds   = as.integer(n_folds),
      max_depth = as.integer(max_depth),
      verbose   = FALSE
    ),
    error = function(e) {
      stop(paste("EffectXShift::run_rct_workflow failed:", conditionMessage(e)))
    }
  )

  list(
    v_ate       = as.numeric(result$v_ate),
    vc_ate      = as.numeric(result$vc_ate),
    contrast    = as.numeric(result$contrast),
    v_se        = as.numeric(result$v_se),
    vc_se       = as.numeric(result$vc_se),
    contrast_se = as.numeric(result$contrast_se),
    rule        = as.character(result$rule),
    converged   = isTRUE(result$converged)
  )
}
