## Formatting/estimand gates, without nuisance fitting. Run Rscript tests/test_concrete_horizon.R
source("r_scripts/concrete_bridge.R")
set.seed(59)
n <- 200L
df <- data.frame(T_obs=seq(0.1, 20, length.out=n), event_type=rep(1L,n),
                 A=rep(c(0L,1L), n/2), W1=rnorm(n), W2=rnorm(n), W3=rnorm(n), W4=rnorm(n))
strict_error <- tryCatch({.concrete_args(df, 20, horizon_policy="strict"); FALSE},
                         error=function(e) grepl("Strict RMST horizon", conditionMessage(e)))
stopifnot(strict_error)
accepted <- .concrete_args(df, 12, horizon_policy="strict")
stopifnot(identical(accepted$horizon, 12))
legacy <- .concrete_args(df, 20)
stopifnot(identical(legacy$horizon, 20*0.999))
cat("Strict horizon rejection/preservation and legacy cap gates passed\n")
