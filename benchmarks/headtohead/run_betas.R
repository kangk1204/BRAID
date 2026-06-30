#!/usr/bin/env Rscript
# Real betAS ΔPSI intervals on the exported per-replicate counts.
#
# Faithfully reproduces betAS's estimator using the package's OWN internal
# functions: per replicate PSI ~ Beta(shape1=inc, shape2=exc) (betAS's
# `individualBetas_nofitting_incr`), zero counts replaced by
# `givenCovfindIncr(cov, maxDevSimulationN100)`, per-replicate draws pooled within
# a group, and ΔPSI = pool(ctrl) - pool(treat) (betAS's `jointBetas_nofitting`).
# Outputs a 1-alpha interval per event at several nominal levels via quantiles of
# the ΔPSI distribution -- the same quantity every other method reports.

suppressMessages(library(betAS))
ns <- getNamespace("betAS")
givenCovfindIncr <- get("givenCovfindIncr", envir = ns)
maxdevRef <- get("maxDevSimulationN100", envir = ns)

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[[1]]
truth_path  <- args[[2]]
out_path    <- args[[3]]
npoints <- if (length(args) >= 4) as.integer(args[[4]]) else 2000L

counts <- read.delim(counts_path, stringsAsFactors = FALSE)
truth  <- read.delim(truth_path, stringsAsFactors = FALSE)

levels <- c(0.50, 0.80, 0.90, 0.95, 0.99)

safe_incr <- function(cov) {
  # betAS's increment lookup (maxDevSimulationN100) only spans coverage up to ~100;
  # for higher coverage (cap) or any internal failure, fall back to a small Jeffreys-
  # like pseudocount so high-coverage zero-count events don't halt the run.
  tryCatch(givenCovfindIncr(cov = min(max(round(cov), 1), 100), maxDevRef = maxdevRef),
           error = function(e) 0.5)
}

group_pool <- function(sub) {
  # sub: data.frame rows for one event+group, columns inc, exc (per replicate)
  draws <- c()
  for (i in seq_len(nrow(sub))) {
    inc <- sub$inc[i]; exc <- sub$exc[i]
    if (inc == 0) inc <- safe_incr(exc)
    if (exc == 0) exc <- safe_incr(inc)
    if (inc <= 0) inc <- 1e-6
    if (exc <= 0) exc <- 1e-6
    draws <- c(draws, rbeta(npoints, shape1 = inc, shape2 = exc))
  }
  draws
}

set.seed(7)
keys <- unique(truth$key)
res <- data.frame(key = keys, stringsAsFactors = FALSE)
res$dpsi_mean <- NA_real_
res$dpsi_std  <- NA_real_
for (lv in levels) {
  res[[sprintf("low_%.2f", lv)]] <- NA_real_
  res[[sprintf("high_%.2f", lv)]] <- NA_real_
}

for (k in keys) {
  cc <- counts[counts$key == k & counts$group == "ctrl", ]
  ct <- counts[counts$key == k & counts$group == "treat", ]
  if (nrow(cc) == 0 || nrow(ct) == 0) next
  poolC <- group_pool(cc)
  poolT <- group_pool(ct)
  m <- min(length(poolC), length(poolT))
  dpsi <- sample(poolC, m) - sample(poolT, m)
  ri <- which(res$key == k)
  res$dpsi_mean[ri] <- mean(dpsi)
  res$dpsi_std[ri]  <- sd(dpsi)
  for (lv in levels) {
    a <- 1 - lv
    res[[sprintf("low_%.2f", lv)]][ri]  <- as.numeric(quantile(dpsi, a / 2))
    res[[sprintf("high_%.2f", lv)]][ri] <- as.numeric(quantile(dpsi, 1 - a / 2))
  }
}

write.table(res, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("wrote %d events -> %s\n", length(keys), out_path))
