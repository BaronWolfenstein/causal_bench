## bcf_bart_bridge.R
##
## rpy2 bridge for BCF/BART CATE estimation with rpart summary tree.
## Sourced from causal_bench/estimators/bcf_bart.py via rpy2.
##
## Implements Hahn's (2026-06-25) proposed alternative to EffectXShift:
##   1. Fit BCF (Hahn et al. 2020) or BART to estimate per-patient CATEs
##   2. Grow a parsimonious rpart tree to the CATE point estimates
##   3. Return posterior mean and SD for the highest leaf, lowest leaf,
##      and their contrast (H - L), derived from MCMC draws
##
## Prefers 'bcf' R package (Bayesian Causal Forest — separates prognostic
## and treatment effect surfaces). Falls back to vanilla 'BART' if bcf
## is unavailable.
##
## run_bcf_bart_bridge(df, outcome_col, treatment_col, covariate_cols,
##                     nburn, nsim, min_leaf_n)
##
## Returns a named list:
##   $high_leaf_ate   numeric — posterior mean ATE in the highest-CATE leaf
##   $high_leaf_se    numeric — posterior SD of high_leaf_ate
##   $low_leaf_ate    numeric — posterior mean ATE in the lowest-CATE leaf
##   $low_leaf_se     numeric — posterior SD of low_leaf_ate
##   $contrast        numeric — high_leaf_ate - low_leaf_ate
##   $contrast_se     numeric — posterior SD of the contrast
##   $n_leaves        integer — number of terminal leaves
##   $rule_high       character — rpart row-name of the high leaf (for debugging)
##   $rule_low        character — rpart row-name of the low leaf
##   $converged       logical — FALSE if tree degenerated (no splits)

.use_bcf <- FALSE

suppressPackageStartupMessages({
  if (requireNamespace("bcf", quietly = TRUE)) {
    library(bcf)
    .use_bcf <- TRUE
  } else if (requireNamespace("BART", quietly = TRUE)) {
    library(BART)
  } else {
    stop(
      "Neither 'bcf' nor 'BART' R package is available. ",
      "Install with: install.packages('bcf')"
    )
  }
  library(rpart)
})

run_bcf_bart_bridge <- function(df,
                                 outcome_col    = "Y",
                                 treatment_col  = "A",
                                 covariate_cols = c("W1", "W2", "W3", "W4"),
                                 nburn          = 500L,
                                 nsim           = 500L,
                                 min_leaf_n     = 10L) {
  stopifnot(is.data.frame(df))
  outcome_col   <- as.character(outcome_col[1])
  treatment_col <- as.character(treatment_col[1])
  stopifnot(outcome_col   %in% names(df))
  stopifnot(treatment_col %in% names(df))
  stopifnot(all(covariate_cols %in% names(df)))

  Y <- as.numeric(df[[outcome_col]])
  A <- as.integer(df[[treatment_col]])
  W <- as.matrix(df[, covariate_cols, drop = FALSE])
  n <- length(Y)

  nburn <- as.integer(nburn)
  nsim  <- as.integer(nsim)

  if (.use_bcf) {
    pihat <- tryCatch(
      predict(glm(A ~ W, family = binomial()), type = "response"),
      error = function(e) rep(mean(A), n)
    )
    fit <- bcf(
      y          = Y,
      z          = A,
      x_control  = W,
      x_moderate = W,
      pihat      = pihat,
      nburn      = nburn,
      nsim       = nsim,
      verbose    = FALSE
    )
    # fit$tau is (nsim × n)
    tau_post <- fit$tau
  } else {
    Xtrain  <- cbind(W, A)
    bart_fit <- wbart(
      x.train   = Xtrain,
      y.train   = Y,
      ndpost    = nsim,
      nskip     = nburn,
      printevery = 0L
    )
    X1 <- cbind(W, rep(1L, n))
    X0 <- cbind(W, rep(0L, n))
    y1_pred  <- predict(bart_fit, X1)   # (nsim × n)
    y0_pred  <- predict(bart_fit, X0)
    tau_post <- y1_pred - y0_pred
  }

  tau_hat <- colMeans(tau_post)   # point estimates (length n)

  # Summary tree: rpart on CATE point estimates
  cate_df <- as.data.frame(W)
  colnames(cate_df) <- covariate_cols
  cate_df$.tau <- tau_hat

  tree <- rpart(
    .tau ~ .,
    data    = cate_df,
    method  = "anova",
    control = rpart.control(
      cp        = 0.01,
      minsplit  = as.integer(min_leaf_n) * 2L,
      minbucket = as.integer(min_leaf_n)
    )
  )

  leaf_ids <- unique(tree$where)
  if (length(leaf_ids) < 2L) {
    return(list(
      high_leaf_ate = NA_real_, high_leaf_se = NA_real_,
      low_leaf_ate  = NA_real_, low_leaf_se  = NA_real_,
      contrast      = NA_real_, contrast_se  = NA_real_,
      n_leaves      = 1L,
      rule_high     = "no_split", rule_low = "no_split",
      top_split_var = "no_split",
      converged     = FALSE
    ))
  }

  # Variable used at the root split — the primary subgroup boundary.
  top_split_var <- as.character(tree$frame$var[1])

  leaf_means <- tapply(tau_hat, tree$where, mean)
  high_id    <- as.integer(names(which.max(leaf_means)))
  low_id     <- as.integer(names(which.min(leaf_means)))

  hi_mask <- tree$where == high_id
  lo_mask <- tree$where == low_id

  # Leaf-level posterior CATE draws: average over patients in each leaf
  hi_draws   <- rowMeans(tau_post[, hi_mask, drop = FALSE])  # length nsim
  lo_draws   <- rowMeans(tau_post[, lo_mask, drop = FALSE])
  cont_draws <- hi_draws - lo_draws

  list(
    high_leaf_ate = mean(hi_draws),
    high_leaf_se  = sd(hi_draws),
    low_leaf_ate  = mean(lo_draws),
    low_leaf_se   = sd(lo_draws),
    contrast      = mean(cont_draws),
    contrast_se   = sd(cont_draws),
    n_leaves      = length(leaf_ids),
    rule_high     = as.character(rownames(tree$frame)[high_id]),
    rule_low      = as.character(rownames(tree$frame)[low_id]),
    top_split_var = top_split_var,
    converged     = TRUE
  )
}
