# Plot combined model predictions from gestational age-specific models
library(optparse)
library(logger)
library(dplyr)
library(readr)
library(data.table)
library(tidyr)
library(caret)
library(pROC)
library(grid)
library(gridExtra)
library(yardstick)
library(assertthat)
library(GGally)
library(ggthemes)
library(ggplot2)

theme_set(theme_base(base_size=24))

# Read in results from each gestational age model
results_dir <- "./results/deep_mtl/neonatal_bottleneck_validation_per_gestage/"
ga_specific_model_subdirs <- c(
	"22_23/ensemble_bottle_10/", "24_25/ensemble_bottle_10/", "26_27/ensemble_bottle_10/", "28_29/ensemble_bottle_10/"
)

# Read in outcomes for matching and calculating aupr
neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt") %>% sort()
true_vals <- fread("./data/processed/neonatal_conditions.csv") %>%
	select(row_id, one_of(neonatal_outcomes))
true_vals[, total_conditions:=rowSums(.SD), .SDcols=neonatal_outcomes]
true_vals_tall <- gather(
	true_vals, "neonatal_outcome", "indicator", -total_conditions, -row_id)
outcomes_colors <- tableau_color_pal("Tableau 10")(length(neonatal_outcomes))
names(outcomes_colors) <- neonatal_outcomes

# Combine in one dataframe
all_dfs <- list()
prediction_cols <- gsub("_any", "_pred", neonatal_outcomes)
for (model_subdir in ga_specific_model_subdirs) {
	model_file <- paste0(results_dir, model_subdir, "valid_preds.csv")
	model_results <- fread(model_file)
	setnames(model_results, "V1", "row_id")
	model_results <- model_results %>% select(-epoch, -fold, -iter) %>%
		group_by(row_id) %>% summarize_all(mean)
	model_results[["ga_model"]] <- gsub("/ensemble_bottle_10/", "", model_subdir)
	all_dfs <- append(list(model_results), all_dfs)
}
results_df <- bind_rows(all_dfs)

# Read in metadata for gestational age
metadata <- fread("./data/processed/metadata.csv") %>%
	select(row_id, gacat)

# For each GA range, take predictions from that model weighted with
# predictions from models that are close by.
model_preds_only <- list(
	"22_23"=c(1.0, 0, 0, 0),
	"24_25"=c(0, 1.0, 0, 0),
	"26_27"=c(0, 0, 1.0, 0),
	"28_29"=c(0, 0, 0, 1.0)
)

ga_ranges <- results_df[["ga_model"]] %>% unique() %>% sort()
pdf("./results/deep_mtl/neonatal_bottleneck_validation_per_gestage/combined_preds_curves_model_outputs.pdf",
		height=7, width=7, useDingbats=FALSE)
for (ga in ga_ranges) {
	row_ids_within_ga <- metadata %>% filter(gacat == ga)
	ga_model_preds <- filter(results_df, ga_model == ga) %>%
		filter(row_id %in% row_ids_within_ga[["row_id"]]) %>%
		gather("neonatal_outcome", "pred", -row_id, -ga_model)
	other_model_preds <- filter(results_df, ga_model != ga) %>%
		filter(row_id %in% ga_model_preds[["row_id"]]) %>%
		gather("neonatal_outcome", "pred", -row_id, -ga_model)
	model_preds_for_eval <- ga_model_preds
	#model_preds_for_eval <- bind_rows(ga_model_preds, other_model_preds)
	#model_preds_for_eval <- merge(ga_model_preds, other_model_preds) %>%
	#	select(row_id, all_of(ga_ranges)) %>%
	#	filter(row_id %in% row_ids_within_ga[["row_id"]])


	# Plot model predictions current gestational age specific model
	score_plot_colors <- c(outcomes_colors, c(any_condition="#B07AA1", none="light grey"))
	names(score_plot_colors) <- gsub("_any$", "", names(score_plot_colors))
	score_df <- merge(ga_model_preds, true_vals_tall, by=c("row_id", "neonatal_outcome")) %>%
		mutate(any_condition=ifelse(total_conditions >= 1, 1, 0)) %>%
		mutate(indicator=factor(indicator)) %>%
		mutate(outcome_indicator=case_when(
			indicator == 1 ~ gsub("_any$", "", neonatal_outcome),
			indicator == 0 & total_conditions == 0 ~ "none",
			indicator == 0 & total_conditions >= 1 ~ "other_conditions",
			TRUE ~ "uncategorized")) %>%
		filter(!(outcome_indicator %in% c("other_conditions", "uncategorized")))
	plt <- ggplot(score_df, aes(indicator, pred)) +
		geom_jitter(aes(fill=outcome_indicator), shape=21, color="black", alpha=.4, width=0.3) +
		geom_boxplot(aes(fill=outcome_indicator), alpha=0.7, outlier.shape=NA) +
		scale_fill_manual(values=score_plot_colors, name="Outcome") +
		facet_wrap(~ neonatal_outcome, ncol=3) +
		ggtitle(paste0(ga, " model prediction probabilities per condition")) +
		theme(legend.position="none", plot.title=element_text(size=18))
	print(plt)

	# Plot precision-recall curves
	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	for (outcome in neonatal_outcomes) {
		df_for_roc <- score_df %>% filter(neonatal_outcome == outcome)
		assertion <- assert_that(
			dim(df_for_roc)[1] == df_for_roc[["row_id"]] %>% unique() %>% length())

		event_level_to_use <- "second"
		pr_df <- pr_curve(
			df_for_roc, indicator, pred, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc, indicator, pred, event_level=event_level_to_use)

		pr_df[["Neonatal\nOutcome"]] <- outcome
		prcurve_list[[outcome]] <- pr_df
		aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
		aupr_list[[outcome]] <- aupr_label
	}
	combined_plt <- ggplot(bind_rows(prcurve_list), aes(x=Recall, y=Precision, color=`Neonatal\nOutcome`)) +
		geom_path() +
		scale_color_tableau() +
		ylim(0, 1) +
		annotate("text", x=0.5, y=0.3, size=6, color=outcomes_colors[["bpd_any"]],
						label=paste("BPD", aupr_list[["bpd_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.25, size=6, color=outcomes_colors[["ivh_any"]],
						label=paste("IVH", aupr_list[["ivh_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.2, size=6, color=outcomes_colors[["nec_any"]],
						label=paste("NEC", aupr_list[["nec_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.15, size=6, color=outcomes_colors[["rop_any"]],
						label=paste("ROP", aupr_list[["rop_any"]]), hjust=0) +
		ggtitle(paste0(ga, " model prediction on within-GA set")) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1, plot.title=element_text(size=18))
	print(combined_plt)
}

# TODO: Get model predictions from other models for one gestational age range
# Look at 28-29 weeks because performance is particularly bad.
test_ga <- "28_29"
row_ids_within_ga <- metadata %>% filter(gacat == ga)
ga_model_preds <- filter(results_df, ga_model == test_ga) %>%
	filter(row_id %in% row_ids_within_ga[["row_id"]]) %>%
	gather("neonatal_outcome", "pred", -row_id, -ga_model)
other_model_preds <- filter(results_df, ga_model != test_ga) %>%
	filter(row_id %in% ga_model_preds[["row_id"]]) %>%
	gather("neonatal_outcome", "pred", -row_id, -ga_model)
model_preds_for_eval <- bind_rows(ga_model_preds, other_model_preds)

other_ga <- ga_ranges[ga_ranges != test_ga]
for (ga_subset in ga_ranges) {
	score_df <- merge(
			model_preds_for_eval %>% filter(ga_model == ga_subset),
			true_vals_tall, by=c("row_id", "neonatal_outcome")) %>%
		mutate(any_condition=ifelse(total_conditions >= 1, 1, 0)) %>%
		mutate(indicator=factor(indicator)) %>%
		mutate(outcome_indicator=case_when(
			indicator == 1 ~ gsub("_any$", "", neonatal_outcome),
			indicator == 0 & total_conditions == 0 ~ "none",
			indicator == 0 & total_conditions >= 1 ~ "other_conditions",
			TRUE ~ "uncategorized")) %>%
		filter(!(outcome_indicator %in% c("other_conditions", "uncategorized")))
	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	for (outcome in neonatal_outcomes) {
		df_for_roc <- score_df %>% filter(neonatal_outcome == outcome)
		assertion <- assert_that(
			dim(df_for_roc)[1] == df_for_roc[["row_id"]] %>% unique() %>% length())

		event_level_to_use <- "second"
		pr_df <- pr_curve(
			df_for_roc, indicator, pred, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc, indicator, pred, event_level=event_level_to_use)

		pr_df[["Neonatal\nOutcome"]] <- outcome
		prcurve_list[[outcome]] <- pr_df
		aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
		aupr_list[[outcome]] <- aupr_label
	}
	combined_plt <- ggplot(bind_rows(prcurve_list), aes(x=Recall, y=Precision, color=`Neonatal\nOutcome`)) +
		geom_path() +
		scale_color_tableau() +
		ylim(0, 1) +
		annotate("text", x=0.5, y=0.3, size=6, color=outcomes_colors[["bpd_any"]],
						label=paste("BPD", aupr_list[["bpd_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.25, size=6, color=outcomes_colors[["ivh_any"]],
						label=paste("IVH", aupr_list[["ivh_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.2, size=6, color=outcomes_colors[["nec_any"]],
						label=paste("NEC", aupr_list[["nec_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.15, size=6, color=outcomes_colors[["rop_any"]],
						label=paste("ROP", aupr_list[["rop_any"]]), hjust=0) +
		ggtitle(paste0(ga_subset, " model prediction on ", test_ga)) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1, plot.title=element_text(size=18))
	print(combined_plt)
}
dev.off()

