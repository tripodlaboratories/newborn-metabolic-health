# Plot combined bottleneck outputs from gestational age-specific models
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
	"22_23/ensemble_bottle_1/", "24_25/ensemble_bottle_1/", "26_27/ensemble_bottle_1/", "28_29/ensemble_bottle_1/"
)

# Combine in one dataframe
all_dfs <- list()
for (model_subdir in ga_specific_model_subdirs) {
	model_file <- paste0(results_dir, model_subdir, "valid_bottleneck.csv")
	model_results <- fread(model_file)
	setnames(model_results, "V1", "row_id")
	model_results <- model_results %>% select(-epoch, -fold, -iter) %>%
		group_by(row_id) %>% summarize(mean_bottleneck=mean(bottleneck_unit_0))
	model_results[["ga_model"]] <- gsub("/ensemble_bottle_1/", "", model_subdir)
	all_dfs <- append(list(model_results), all_dfs)
}
results_df <- bind_rows(all_dfs)

# Read in outcomes for matching and calculating aupr
neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt") %>% sort()
true_vals <- fread("./data/processed/neonatal_conditions.csv") %>%
	select(row_id, one_of(neonatal_outcomes))
true_vals[, total_conditions:=rowSums(.SD), .SDcols=neonatal_outcomes]
outcomes_colors <- tableau_color_pal("Tableau 10")(length(neonatal_outcomes))
names(outcomes_colors) <- neonatal_outcomes

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
# TODO: Test a set of weights for 26-27 weeks.
ga_weights <- list(
	"22_23"=list("bpd_any"=c(1.0, 0.2, 0.1, 0.05),
							 "ivh_any"=c(1.0, 0.2, 0.1, 0.05),
							 "nec_any"=c(1.0, 0.7, 0.4, 0.1),
							 "rop_any"=c(1.0, 0.2, 0.1, 0.05)),
	"24_25"=list("bpd_any"=c(0.2, 1.0, 0.2, 0.05),
							 "ivh_any"=c(0.2, 1.0, 0.2, 0.05),
							 "nec_any"=c(0.7, 1.0, 0.7, 0.4),
							 "rop_any"=c(0.2, 1.0, 0.2, 0.05)),
	"26_27"=list("bpd_any"=c(0.33, 0.33, 0.33, 0.01),
							 "ivh_any"=c(0.33, 0.33, 0.33, 0.01),
							 "nec_any"=c(0.01, 0.01, 0.01, 0.97),
							 "rop_any"=c(0.33, 0.33, 0.33, 0.01)),
	"28_29"=list("bpd_any"=c(0.05, 0.1, 0.2, 1.0),
							 "ivh_any"=c(0.05, 0.1, 0.2, 1.0),
							 "nec_any"=c(0.2, 0.4, 0.7, 1.0),
							 "rop_any"=c(0.05, 0.1, 0.2, 1.0))
)
test_weights_26_27 <- list(
	"bpd_any"=c(0, 0, 1, 0),
	"ivh_any"=c(0, 0, 1, 0),
	"nec_any"=c(0, 0, 0, 1),
	"rop_any"=c(0, 0, 1, 0))

# If the bottleneck unit is higher in controls - have to reverse
# the direction for AUPR/AUROC calculation.
preds_dirs <- c(
	"22_23"="forward",
	"24_25"="forward",
	"26_27"="forward",
	"28_29"="forward")

ga_ranges <- results_df[["ga_model"]] %>% unique() %>% sort()
pdf("./results/deep_mtl/neonatal_bottleneck_validation_per_gestage/combined_preds_curves.pdf",
		height=7, width=7, useDingbats=FALSE)

# TODO: Plot weights
#weights_df <- as_tibble(ga_weights)
#weights_df[["ga_model"]] <- ga_ranges
#weights_df <- weights_df %>% gather("ga_range", "weight", -ga_model)
#plt <- ggplot(weights_df, aes(x=ga_range, y=weight)) +
#	geom_col(fill="white", color="black") +
#	facet_wrap(~ga_model, ncol=2) +
#	theme(aspect.ratio=1.4, axis.text.x=element_text(size=11), axis.text.y=element_text(size=11))
#print(plt)

for (ga in ga_ranges) {
	ga_model_preds <- filter(results_df, ga_model == ga) %>%
		spread(ga_model, mean_bottleneck)
	other_model_preds <- filter(results_df, ga_model != ga) %>%
		filter(row_id %in% ga_model_preds[["row_id"]]) %>%
		spread(ga_model, mean_bottleneck) %>%
		drop_na()
	model_preds_for_eval <- merge(ga_model_preds, other_model_preds) %>%
		select(row_id, one_of(ga_ranges))
	model_weights <- ga_weights[[ga]]

	# Plot precision-recall curves
	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	for (outcome in neonatal_outcomes) {
		# Use a different set of model weights for each outcome - the idea is that
		# if models are good at learning one specific outcome within one GA, we
		# don't need the contributions from the other models
		weight_matrix <- matrix(
			model_weights[[outcome]], nrow=nrow(model_preds_for_eval),
			ncol=length(model_weights), byrow=TRUE)
		model_preds_for_eval <- model_preds_for_eval %>%
			mutate(weighted_combination=(model_preds_for_eval[, ga_ranges] * weight_matrix) %>% rowSums())
		preds_true_vals <- merge(model_preds_for_eval, true_vals, by="row_id")
			df_for_roc <- preds_true_vals %>%
				filter(!(get(outcome) == 0 & total_conditions >= 1))
		preds_dir <- preds_dirs[[ga]]
		if (preds_dir == "reverse") {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}

		pr_df <- pr_curve(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use)

		pr_df[["Neonatal\nOutcome"]] <- outcome
		prcurve_list[[outcome]] <- pr_df
		aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
		aupr_list[[outcome]] <- aupr_label
		plt <- ggplot(pr_df, aes(x=Recall, y=Precision)) +
			geom_path(color=outcomes_colors[[outcome]]) +
			ylim(0, 1) +
			annotate("text", x=0.8, y=1.0, size=10, label=aupr_label) +
			ggtitle(paste(outcome, "PR-Curve")) +
			theme(aspect.ratio=1)
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
		ggtitle(paste0(ga, " model w/weighted combination")) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1, plot.title=element_text(size=18))
	print(combined_plt)

	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	ga_model_preds_only <- model_preds_only[[ga]]
	for (outcome in neonatal_outcomes) {
		# Use a different set of model weights for each outcome - the idea is that
		# if models are good at learning one specific outcome within one GA, we
		# don't need the contributions from the other models
		weight_matrix <- matrix(
			ga_model_preds_only, nrow=nrow(model_preds_for_eval),
			ncol=length(model_weights), byrow=TRUE)
		model_preds_for_eval <- model_preds_for_eval %>%
			mutate(weighted_combination=(model_preds_for_eval[, ga_ranges] * weight_matrix) %>% rowSums())
		preds_true_vals <- merge(model_preds_for_eval, true_vals, by="row_id")
			df_for_roc <- preds_true_vals %>%
				filter(!(get(outcome) == 0 & total_conditions >= 1))
		preds_dir <- preds_dirs[[ga]]
		if (preds_dir == "reverse") {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}

		pr_df <- pr_curve(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use)

		pr_df[["Neonatal\nOutcome"]] <- outcome
		prcurve_list[[outcome]] <- pr_df
		aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
		aupr_list[[outcome]] <- aupr_label
		plt <- ggplot(pr_df, aes(x=Recall, y=Precision)) +
			geom_path(color=outcomes_colors[[outcome]]) +
			ylim(0, 1) +
			annotate("text", x=0.8, y=1.0, size=10, label=aupr_label) +
			ggtitle(paste(outcome, "PR-Curve")) +
			theme(aspect.ratio=1)
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
		ggtitle(paste0(ga, "(no contribution from other models)")) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1, plot.title=element_text(size=18))
	print(combined_plt)
}
# Repeat where predictions are kept only for validation samples of the
# same gestational age.
for (ga in ga_ranges) {
	row_ids_within_ga <- metadata %>% filter(gacat == ga)
	ga_model_preds <- filter(results_df, ga_model == ga) %>%
		spread(ga_model, mean_bottleneck)
	other_model_preds <- filter(results_df, ga_model != ga) %>%
		filter(row_id %in% ga_model_preds[["row_id"]]) %>%
		spread(ga_model, mean_bottleneck) %>%
		drop_na()
	model_preds_for_eval <- merge(ga_model_preds, other_model_preds) %>%
		select(row_id, all_of(ga_ranges)) %>%
		filter(row_id %in% row_ids_within_ga[["row_id"]])
	model_weights <- ga_weights[[ga]]

	# Plot model predictions current gestational age specific model
	score_plot_colors <- c(outcomes_colors, c(any_condition="#B07AA1", none="light grey"))
	names(score_plot_colors) <- gsub("_any$", "", names(score_plot_colors))
	score_df <- merge(model_preds_for_eval, true_vals, by="row_id") %>%
		mutate(any_condition=ifelse(total_conditions >= 1, 1, 0)) %>%
		select(!!ga, one_of(neonatal_outcomes), any_condition, total_conditions) %>%
		gather("neonatal_outcome", "indicator", -!!ga, -total_conditions) %>%
		rename(gestage_model_output=ga) %>%
		mutate(indicator=factor(indicator)) %>%
		mutate(outcome_indicator=case_when(
			indicator == 1 ~ gsub("_any$", "", neonatal_outcome),
			indicator == 0 & total_conditions == 0 ~ "none",
			indicator == 0 & total_conditions >= 1 ~ "other_conditions",
			TRUE ~ "uncategorized")) %>%
		filter(!(outcome_indicator %in% c("other_conditions", "uncategorized")))
	plt <- ggplot(score_df, aes(indicator, gestage_model_output)) +
		geom_jitter(aes(fill=outcome_indicator), shape=21, color="black", alpha=.4, width=0.3) +
		geom_boxplot(aes(fill=outcome_indicator), alpha=0.7, outlier.shape=NA) +
		scale_fill_manual(values=score_plot_colors, name="Outcome") +
		facet_wrap(~ neonatal_outcome, ncol=3) +
		ggtitle(paste0(ga, " model outputs per condition")) +
		theme(legend.position="none", plot.title=element_text(size=18))
	print(plt)

	# Plot precision-recall curves
	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	for (outcome in neonatal_outcomes) {
		weight_matrix <- matrix(
			model_weights[[outcome]], nrow=nrow(model_preds_for_eval),
			ncol=length(model_weights), byrow=TRUE)
		model_preds_for_outcome_eval <- model_preds_for_eval %>%
			mutate(weighted_combination=(model_preds_for_eval[, ga_ranges] * weight_matrix) %>% rowSums())
		preds_true_vals <- merge(model_preds_for_outcome_eval, true_vals, by="row_id")
		df_for_roc <- preds_true_vals %>%
			filter(!(get(outcome) == 0 & total_conditions >= 1))

		preds_dir <- preds_dirs[[ga]]
		if (preds_dir == "reverse") {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}

		pr_df <- pr_curve(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use)

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
		ggtitle(paste0(ga, " model w/weighted combination")) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1, plot.title=element_text(size=18))
	print(combined_plt)

	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	ga_model_preds_only <- model_preds_only[[ga]]
	for (outcome in neonatal_outcomes) {
		# Use a different set of model weights for each outcome - the idea is that
		# if models are good at learning one specific outcome within one GA, we
		# don't need the contributions from the other models
		weight_matrix <- matrix(
			ga_model_preds_only, nrow=nrow(model_preds_for_eval),
			ncol=length(model_weights), byrow=TRUE)
		model_preds_for_eval <- model_preds_for_eval %>%
			mutate(weighted_combination=(model_preds_for_eval[, ga_ranges] * weight_matrix) %>% rowSums())
		preds_true_vals <- merge(model_preds_for_eval, true_vals, by="row_id")
			df_for_roc <- preds_true_vals %>%
				filter(!(get(outcome) == 0 & total_conditions >= 1))
		preds_dir <- preds_dirs[[ga]]
		if (preds_dir == "reverse") {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}

		pr_df <- pr_curve(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, weighted_combination, event_level=event_level_to_use)

		pr_df[["Neonatal\nOutcome"]] <- outcome
		prcurve_list[[outcome]] <- pr_df
		aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
		aupr_list[[outcome]] <- aupr_label
		plt <- ggplot(pr_df, aes(x=Recall, y=Precision)) +
			geom_path(color=outcomes_colors[[outcome]]) +
			ylim(0, 1) +
			annotate("text", x=0.8, y=1.0, size=10, label=aupr_label) +
			ggtitle(paste(outcome, "PR-Curve")) +
			theme(aspect.ratio=1)
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
		ggtitle(paste0(ga, "(no contribution from other models)")) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1, plot.title=element_text(size=18))
	print(combined_plt)
}

# We know that performance is not as good on 26-27 weeks, but how well do
# the other models perform?
ga <- "26_27"
row_ids_within_ga <- metadata %>% filter(gacat == ga)
ga_model_preds <- filter(results_df, ga_model == ga) %>%
	spread(ga_model, mean_bottleneck)
other_model_preds <- filter(results_df, ga_model != ga) %>%
	filter(row_id %in% ga_model_preds[["row_id"]]) %>%
	spread(ga_model, mean_bottleneck) %>%
	drop_na()
model_preds_for_eval <- merge(ga_model_preds, other_model_preds) %>%
	select(row_id, one_of(ga_ranges)) %>%
	filter(row_id %in% row_ids_within_ga[["row_id"]])

other_ga_ranges <- ga_ranges[ga_ranges != ga]
for (other_ga in other_ga_ranges) {
	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	preds_from_other_ga <- model_preds_for_eval %>%
		select(row_id, !!other_ga)
	for (outcome in neonatal_outcomes) {
		preds_true_vals <- merge(preds_from_other_ga, true_vals, by="row_id")
		df_for_roc <- preds_true_vals %>%
			filter(!(get(outcome) == 0 & total_conditions >= 1))
		preds_dir <- preds_dirs[[other_ga]]
		if (preds_dir == "reverse") {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}

		pr_df <- pr_curve(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, !!other_ga, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, !!other_ga, event_level=event_level_to_use)

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
		ggtitle(paste0(other_ga, " model prediction on ", ga, " samples")) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1, plot.title=element_text(size=18))
	print(combined_plt)
}

# TODO: Make predictions using weights for 26-27 weeks of GA
#model_weights <- ga_weights[[ga]]
model_weights <- test_weights_26_27

prcurve_list <- vector("list", length(neonatal_outcomes))
names(prcurve_list) <- neonatal_outcomes
aupr_list <- vector("character", length(neonatal_outcomes))
names(aupr_list) <- neonatal_outcomes

# Range standardization function designed to act on a vector level
sigmoid_transform <- function(x) {
	x <- scale(x) %>% as.vector()
	1 / (1 + exp(-x))
}

sigmoid_preds <- model_preds_for_eval
sigmoid_preds[ga_ranges] <- lapply(sigmoid_preds[ga_ranges], sigmoid_transform)
cols_to_invert <- preds_dirs[preds_dirs == "forward"] %>% names()
sigmoid_preds[cols_to_invert] <- lapply(sigmoid_preds[cols_to_invert], function(x) {1 - x})

for (outcome in neonatal_outcomes) {
	# TODO: Figure out how to get directionality of predictions
	weight_matrix <- matrix(
		model_weights[[outcome]], nrow=nrow(model_preds_for_eval),
		ncol=length(model_weights), byrow=TRUE)
	model_preds_for_outcome_eval <- sigmoid_preds %>%
		mutate(weighted_combination=(sigmoid_preds[, ga_ranges] * weight_matrix) %>% rowSums())
	preds_true_vals <- merge(model_preds_for_outcome_eval, true_vals, by="row_id")
	df_for_roc <- preds_true_vals %>%
		filter(!(get(outcome) == 0 & total_conditions >= 1))

	roc_preds_df <- df_for_roc %>%
		mutate(any_condition=ifelse(total_conditions >= 1, 1, 0)) %>%
		select(weighted_combination, !!outcome) %>%
		mutate(!!as.name(outcome):=factor(!!as.name(outcome))) %>%
		rename(weighted_model_output=weighted_combination)
	plt <- ggplot(roc_preds_df, aes_string(outcome, "weighted_model_output", fill=outcome)) +
		geom_jitter(shape=21, color="black", alpha=.4, width=0.3) +
		geom_boxplot(alpha=0.7, outlier.shape=NA) +
		scale_fill_manual(values=c("0"="light grey", "1"=outcomes_colors[[outcome]])) +
		ggtitle(paste0("predictions for ", outcome)) +
		theme(legend.position="none", plot.title=element_text(size=18))
	print(plt)

	event_level_to_use <- "first"
	pr_df <- pr_curve(
		df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
		!!outcome, weighted_combination, event_level=event_level_to_use) %>%
		rename(Recall=recall, Precision=precision)
	pr_auc_value <- pr_auc(
		df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
		!!outcome, weighted_combination, event_level=event_level_to_use)
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
	ggtitle(paste0("Test weighted prediction on ", ga, " samples")) +
	labs(color="Neonatal\nOutcome") +
	theme(aspect.ratio=1, plot.title=element_text(size=18))
print(combined_plt)

# TODO: Plot sigmoid transformed model outputs for each gestational age model
# within the gestational age and then applied on other gestational age ranges
for (ga in ga_ranges) {
	ga_model_preds <- filter(results_df, ga_model == ga) %>%
		spread(ga_model, mean_bottleneck)
	ga_subset_order <- c(ga, ga_ranges[ga_ranges != ga])

	for (ga_subset in ga_subset_order) {
		row_ids_in_subset <- metadata %>% filter(gacat == ga_subset)
		model_preds_for_eval <- ga_model_preds %>%
			filter(row_id %in% row_ids_in_subset[["row_id"]])
		sigmoid_preds <- model_preds_for_eval %>%
			mutate(!!as.name(ga):=sigmoid_transform(!!as.name(ga)))
		# TODO: Figure out better logic on which columns to invert now that the
		# correct event level is "second"
		cols_to_invert <- c("28_29")
		if (any(colnames(ga_model_preds) %in% cols_to_invert)) {
			col_subset <- intersect(colnames(ga_model_preds), cols_to_invert)
			sigmoid_preds <- sigmoid_preds %>%
				mutate_at(col_subset, function(x) {1 - x})
		}
		# Plot model predictions current gestational age specific model
		preds_true_vals <- merge(sigmoid_preds, true_vals, by="row_id") %>%
			mutate(any_condition=ifelse(total_conditions >= 1, 1, 0))
		score_df <- preds_true_vals %>%
			select(!!ga, one_of(neonatal_outcomes), any_condition, total_conditions) %>%
			gather("neonatal_outcome", "indicator", -!!ga, -total_conditions) %>%
			rename(gestage_model_output=ga) %>%
			mutate(indicator=factor(indicator)) %>%
			mutate(outcome_indicator=case_when(
				indicator == 1 ~ gsub("_any$", "", neonatal_outcome),
				indicator == 0 & total_conditions == 0 ~ "none",
				indicator == 0 & total_conditions >= 1 ~ "other_conditions",
				TRUE ~ "uncategorized")) %>%
			filter(!(outcome_indicator %in% c("other_conditions", "uncategorized")))
		plt <- ggplot(score_df, aes(indicator, gestage_model_output)) +
			geom_jitter(aes(fill=outcome_indicator), shape=21, color="black", alpha=.4, width=0.3) +
			geom_boxplot(aes(fill=outcome_indicator), alpha=0.7, outlier.shape=NA) +
			scale_fill_manual(values=score_plot_colors, name="Outcome") +
			facet_wrap(~ neonatal_outcome, ncol=3) +
			ggtitle(paste0(ga, " model predictions on ", ga_subset)) +
			theme(legend.position="none", plot.title=element_text(size=16))
		print(plt)

		prcurve_list <- vector("list", length(neonatal_outcomes))
		names(prcurve_list) <- neonatal_outcomes
		aupr_list <- vector("character", length(neonatal_outcomes))
		names(aupr_list) <- neonatal_outcomes
		for (outcome in neonatal_outcomes) {
			df_for_roc <- preds_true_vals %>%
				filter(!(get(outcome) == 0 & total_conditions >= 1)) %>%
				mutate(!!as.name(outcome) := factor(!!as.name(outcome)))

			event_level_to_use <- "second"
			pr_df <- pr_curve(
				df_for_roc, !!outcome, !!ga, event_level=event_level_to_use) %>%
				rename(Recall=recall, Precision=precision)
			pr_auc_value <- pr_auc(
				df_for_roc, !!outcome, !!ga, event_level=event_level_to_use)
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
			ggtitle(paste0(ga, " model sigmoid predictions on ", ga_subset)) +
			labs(color="Neonatal\nOutcome") +
			theme(aspect.ratio=1, plot.title=element_text(size=18))
		print(combined_plt)
	}
}

dev.off()

