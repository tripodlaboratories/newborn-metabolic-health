# Plot external validation subgroup discovery results
library(readr)
library(data.table)
library(dplyr)
library(yardstick)
library(ggthemes)
library(ggplot2)

# TODO: Read in results
results_dir <- "./results/external_validation/mednax/"
cal_subgroup_preds <- fread(
	paste0(results_dir, "calbiobank_subgroup_kfold_preds.csv"))

# TODO: Plot the ability of the subgroup discovery KFold predictions to identify
# individuals in the top subgroup
mean_subgroup_preds <- cal_subgroup_preds %>%
	group_by(row_id) %>% select(-iter) %>% summarize_all(mean) %>%
	mutate(isin_top_subgroup=factor(isin_top_subgroup))
cal_roc_curve <- roc_curve(
	mean_subgroup_preds, isin_top_subgroup, topk_prob, event_level="second")
cal_roc_auc <- roc_auc(
	mean_subgroup_preds, isin_top_subgroup, topk_prob, event_level="second")

cal_pr_curve <- pr_curve(
	mean_subgroup_preds, isin_top_subgroup, topk_prob, event_level="second")
cal_auprc <- pr_auc(
	mean_subgroup_preds, isin_top_subgroup, topk_prob, event_level="second")
set.seed(101)
cal_random_baseline <- pr_auc_vec(
	mean_subgroup_preds[["isin_top_subgroup"]],
	runif(nrow(mean_subgroup_preds["isin_top_subgroup"])),
	event_level="second")

# TODO: Predict the subgroups in the top 20% of subgroups

# TODO: Read in the mednax subgroup results and mednax prediction results
mednax_subgroup_preds <- fread(paste0(results_dir, "mednax_subgroup_preds.csv"))

# TODO: Determine the probability cutoff for the top 20% ranked prediction
# probabilities, using 0.80 for the quantile function
desired_percentile <- 0.80
cutoff <- quantile(mednax_subgroup_preds[["topk_prob"]], desired_percentile)

# TODO: Filter out the mednax results and write out the mednax top 20%
mednax_top_preds <- mednax_subgroup_preds %>%
	filter(topk_prob > cutoff)

# TODO: Write out PDF of results?
# pdf()
# dev.off()

