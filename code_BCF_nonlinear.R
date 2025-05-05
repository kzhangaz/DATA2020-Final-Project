#---------------------------------------------------------------------------------
# BCF simulation (ATE & CATE metrics) — clean console, ASCII progress
#---------------------------------------------------------------------------------


# -------------------------------------------------------------------
#  BCF simulation – fixed (no masking, paper-level hyper-parameters)
# -------------------------------------------------------------------
options(echo = FALSE, warn = 0)

## 0) Paths ----------------------------------------------------------
setwd("C:/Users/marti/OneDrive/文档/data2020_final_project")   # adjust if needed
source("Aux_functions.R")                                     # g(), rmse_*, ...

save_dir  <- "C:/Users/marti/OneDrive/文档/data2020_final_project"
dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
filename  <- "BCF_Simulation_results"

## 1) Package load (bcf must come first!) ---------------------------
suppressPackageStartupMessages({
  # load *only* what you need before the loop
  for (pkg in c("bcf", "dbarts", "grf", "bayeslm", "dplyr")) {
    if (!require(pkg, character.only = TRUE))
      install.packages(pkg, repos = "https://cloud.r-project.org")
  }
})

##  --- optional extras AFTER we take a handle to bcf() -------------
# keep a safe handle so later packages can’t mask it
BCF_fun <- bcf::bcf

for (extra in c("nnet", "bartBMA", "bcfbma", "BART")) {
  if (!require(extra, character.only = TRUE))
    install.packages(extra, repos = "https://cloud.r-project.org")
}

## 2) Helpers to silence C++ progress spinners ----------------------
quiet <- function(expr) {
  zz <- file(if (.Platform$OS.type == "windows") "NUL" else "/dev/null", "w")
  sink(zz); on.exit({ sink(); close(zz) }, add = TRUE)
  force(expr)
}
silent_bcf <- function(...) quiet(BCF_fun(...))

## 3) Simulation design ---------------------------------------------
num_rep <- 30                 # Monte-Carlo repetitions
sample  <- c(250)               # sample sizes
ncov    <- c(5)                 # number of covariates

tau_str <- c("homogeneous", "heterogeneous")
mu_str  <- c("nonlinear")

#tau_str <- c("heterogeneous", "homogeneous")
#mu_str  <- c("linear", "nonlinear")

total <- length(sample) * length(ncov) *
  length(mu_str) * length(tau_str) * num_rep
cat(">>> Starting simulation (", total, " fits)\n", sep = "")

consolidated_results <- list()
counter <- 0

## 4) Nested loops -------------------------------------------------------------
for (s in seq_along(sample))
  for (k in seq_along(ncov))
    for (m in seq_along(mu_str))
      for (t in seq_along(tau_str))
        for (i in seq_len(num_rep)) {
          
          counter <- counter + 1
          
          # 4a) training data ----------------------------------------------------------
          D <- generate_data(sample[s], ncov[k], tau_str[t], mu_str[m],
                             seed = i)
          
          # 4b) fit BCF quietly --------------------------------------------------------
          tm <- system.time({
            
            fit <- try(
              silent_bcf(
                y          = D$y,
                z          = D$z,
                x_control  = D$x,
                x_moderate = D$x,
                pihat      = D$pihat,
                nburn          = 1000,   # <- paper
                nsim           = 1000,   # <- paper
                ntree_control  = 200,    # prognostic forest (default)
                ntree_moderate = 50,     # <- paper
                base_control   = 0.95, power_control   = 2,   # default BART prior
                base_moderate  = 0.25, power_moderate  = 3,   # <- stronger depth penalty
                verbose        = FALSE
              ),
              silent = TRUE
            )
            
          })
          
          # 4c) derive metrics ---------------------------------------------------------
          if (!inherits(fit, "try-error")) {
            
            tau_hat  <- colMeans(fit$tau)
            ate_true <- mean(D$tau)
            
            ate_draws <- rowMeans(fit$tau)
            ci_ate    <- quantile(ate_draws, c(0.025, 0.975))
            ci_taus   <- apply(fit$tau, 2, quantile, c(0.025, 0.975))
            
            metrics <- list(
              Algorithm      = "BCF",
              RMSE_CATE      = rmse_cate(D$tau, tau_hat),
              RMSE_ATE       = rmse_ate (D$tau, tau_hat),
              Coverage_ATE   = as.numeric(ate_true >= ci_ate[1] && ate_true <= ci_ate[2]),
              Length_ATE     = diff(ci_ate),
              Coverage_CATE  = mean(D$tau >= ci_taus[1, ] & D$tau <= ci_taus[2, ]),
              Length_CATE    = mean(ci_taus[2, ] - ci_taus[1, ])
            )
            
          } else {
            metrics <- list(Algorithm = "BCF",
                            RMSE_CATE = NA, RMSE_ATE = NA,
                            Coverage_ATE = NA, Length_ATE = NA,
                            Coverage_CATE = NA, Length_CATE = NA)
          }
          
          # 4d) assemble one result row -----------------------------------------------
          consolidated_results[[counter]] <- data.frame(
            metrics,
            rep     = i,
            n       = D$n,
            p       = D$p,
            tau_str = tau_str[t],
            mu_str  = mu_str[m],
            stringsAsFactors = FALSE
          )
          
          # 4e) checkpoint -------------------------------------------------------------
          save(consolidated_results,
               file = file.path(save_dir, paste0(filename, ".RData")))
          
          # 4f) progress line ----------------------------------------------------------
          cat(sprintf("[%-4d/%-4d] n=%d p=%d mu=%s tau=%s rep=%d | %.1f s\n",
                      counter, total, D$n, D$p,
                      mu_str[m], tau_str[t], i,
                      unname(tm["elapsed"])))
          flush.console()
        }

## 5) Finish -------------------------------------------------------------------
cat(">>> Simulation finished. Results saved to ", save_dir, "\n", sep = "")
