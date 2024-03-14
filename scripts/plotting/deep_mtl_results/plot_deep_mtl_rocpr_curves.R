# Plot AUROC plots
# Determing model performance in cases where there is not case overlap
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
library(scales)
library(ggthemes)
library(ggplot2)

theme_set(theme_base(base_size=24))

# args
option_list <- list(
  make_option(
		c("-i", "--input_dir"), type="character",
		help="Input results directory with predictions files and true values files."),
	make_option(
		c("-o", "--output_file"), type="character", default=NULL,
		help="Filename to save resulting plot"),
	make_option(
		c("--preds_filename"), type="character", default="preds.csv.gz",
		help="Naming convention for the file containing predictions"),
	make_option(
		c("--true_vals_file"), type="character", default=NULL,
		help="Separate file to match true values from."),
	make_option(
		c("--no_overlap"), action="store_true", default=FALSE,
		help="Remove overlapping cases from analysis"),
	make_option(
		c("--reference_points"), action="store_true", default=FALSE,
		help="add reference points of preterm, extreme preterm, and sga to plots"),
	make_option(
		c("--summary_tables"), action="store_true", default=FALSE,
		help="Add summary tables to plots."),
	make_option(
		c("--rev_preds_direction"), action="store_true", default=FALSE,
		help="Reverse the direction of values used in predictions columns, smaller values are associated with outcome."),
	make_option(
		c("--controls_as_positive"), action="store_true", default=FALSE,
		help="Use controls as the label of interest"),
	make_option(
		c("--compare_prc_vs_random"), action="store_true", default=FALSE,
		help="Add random classifier performance to precision recall curves."),
	make_option(
		c("--test_against_any"), action="store_true", default=FALSE,
		help="Test against any outcome"
	),
	make_option(
		c("--interactive_debug"), type="logical", default=FALSE,
		help="adds default options for debugging interactively"
	)
)
log_threshold(DEBUG)
opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input_dir
output_file <- opts$output_file
preds_filename <- opts$preds_filename
true_vals_file <- opts$true_vals_file
no_overlap <- opts$no_overlap
reference_points <- opts$reference_points
summary_tables <- opts$summary_tables
rev_preds_direction <- opts$rev_preds_direction
add_random_to_prc <- opts$compare_prc_vs_random
controls_as_positive <- opts$controls_as_positive
test_against_any <- opts$test_against_any
interactive_debug <- opts$interactive_debug

if (isTRUE(interactive_debug)) {
	input_dir <- "./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/"
	preds_filename <- "valid_bottleneck.csv"
	true_vals_file <- "./data/processed/neonatal_conditions.csv"
	# Or if testing for the KFold Test Set
	#input_dir <- "./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/"
	#preds_filename <- "bottleneck.csv"
	#true_vals_file <- NULL

	# Other test options
	output_file <- "./.scratch/test_plot.pdf"
	summary_tables <- TRUE
	controls_as_positive <- TRUE
	add_random_to_prc <- TRUE
}

# Read in predictions and true values
neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt") %>% sort()
outcomes_colors <- tableau_color_pal("Tableau 10")(length(neonatal_outcomes))
names(outcomes_colors) <- neonatal_outcomes

preds <- fread(paste0(input_dir, preds_filename))
setnames(preds, "V1", "row_id", skip_absent=TRUE)

# Handling for using one unit bottleneck as predictions
if (isTRUE(grepl("bottleneck", preds_filename))) {
	log_info("Using first bottleneck unit as predictions")
	preds[, (neonatal_outcomes):=bottleneck_unit_0]
}

if (is.null(true_vals_file)) {
	true_vals <- fread(paste0(input_dir, "true_vals.csv"))
	} else {
	true_vals <- fread(true_vals_file)
	true_vals_cols <- c("row_id", neonatal_outcomes)
	true_vals <- true_vals[, ..true_vals_cols]
}
if ("iter" %in% colnames(true_vals)) {
	true_vals <- true_vals[iter == 0]
}

# Merge predictions and true values together
colnames(preds) <- ifelse(
	colnames(preds) %in% neonatal_outcomes, paste0(colnames(preds), "_pred"),
	colnames(preds))
mean_preds <- preds[, lapply(.SD, mean), by=list(row_id),
										.SDcols=paste0(neonatal_outcomes, "_pred")]
true_vals[, total_conditions:=rowSums(.SD), .SDcols=neonatal_outcomes]
true_vals[, outcome_any:=ifelse(total_conditions >= 1, 1, 0)]
merged_results <- merge(mean_preds, true_vals, by="row_id")

sigmoid_transform <- function(x) {
	x <- scale(x) %>% as.vector()
	1 / (1 + exp(-x))
}

if (isTRUE(summary_tables)) {
	if (grepl("valid", preds_filename)) {
		# In the validation data, the same predictions are made in each fold of the
		# data. So, we take the mean across folds in each iteration.
		preds <- preds[
			, lapply(.SD, mean), by=list(row_id, iter),
			.SDcols=paste0(neonatal_outcomes, "_pred")]
	}

	preds_true_vals <- merge(
		preds %>% select(row_id, iter, contains("_pred")) %>%
			gather("outcome", "pred_value", -row_id, -iter) %>%
			mutate(outcome=gsub("_any_pred", "", outcome) %>% toupper()),
		true_vals %>% select(row_id, total_conditions, contains("_any"), -outcome_any) %>%
			gather("outcome", "outcome_value", -row_id, -total_conditions) %>%
			mutate(outcome=gsub("_any", "", outcome) %>% toupper()),
		by=c("row_id", "outcome"))

	get_performance <- function(
		pred_value, outcome_value, total_conditions, metric_fn, rev_preds_direction,
		event_level_to_use) {
		if (isTRUE(rev_preds_direction)) {
			scaled_preds <- scale(pred_value) %>% as.vector()
			sigmoid_preds <- 1 / (1 + exp(-scaled_preds))
			pred_value <- 1 - sigmoid_preds
		}
		results_df <- data.frame(
			pred=pred_value, outcome=outcome_value, total_conditions=total_conditions) %>%
		filter(!(outcome == 0 & total_conditions > 0)) %>%
		mutate(outcome=factor(outcome)) %>%
		metric_fn(outcome, pred, event_level=event_level_to_use)
		return(results_df$.estimate)
	}

	if (isTRUE(controls_as_positive)) {
		event_level_to_use <- "first"
	} else {
		event_level_to_use <- "second"
	}

	classifier_scores <- preds_true_vals %>%
		group_by(outcome, iter) %>%
		mutate(
			aupr=get_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=pr_auc,
				rev_preds_direction=rev_preds_direction,
				event_level_to_use=event_level_to_use),
			auroc=get_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=roc_auc,
				rev_preds_direction=rev_preds_direction,
				event_level_to_use=event_level_to_use)) %>%
			select(outcome, iter, aupr, auroc) %>%
			distinct()
	performance_summary <- classifier_scores %>%
		gather("curve_type", "auc", -outcome, -iter) %>%
		ungroup() %>%
		group_by(curve_type, outcome) %>%
		dplyr::summarize(mean_auc=mean(auc), sd_auc=sd(auc))

	# Repeat for combined scores
	combined_classifier_scores <- preds_true_vals %>% select(-outcome) %>%
		group_by(iter) %>%
		mutate(
				aupr=get_performance(
					pred_value, outcome_value, total_conditions,
					metric_fn=pr_auc,
					rev_preds_direction=rev_preds_direction,
					event_level_to_use=event_level_to_use),
				auroc=get_performance(
					pred_value, outcome_value, total_conditions,
					metric_fn=roc_auc,
					rev_preds_direction=rev_preds_direction,
					event_level_to_use=event_level_to_use)) %>%
			select(iter, aupr, auroc) %>%
			distinct()
	combined_performance_summary <- combined_classifier_scores %>%
		gather("curve_type", "auc", -iter) %>%
		ungroup() %>%
		group_by(curve_type) %>%
		dplyr::summarize(mean_auc=mean(auc), sd_auc=sd(auc))
}

if (isTRUE(no_overlap)) {
	no_overlap_results <- merged_results[total_conditions <= 1, ]
	assertion <- assert_that(
		nrow(no_overlap_results) < nrow(merged_results),
		msg="Subset on cases without overlap should be less than total dataset")
	merged_results <- no_overlap_results
}

if (isTRUE(reference_points)) {
	metadata <- fread("./data/processed/metadata.csv")
	metadata <- metadata[, list(row_id, gacat, sga_who, sga_nichd)]
	metadata[, gacat:=gacat %>% gsub("^[0-9]{2}_", "", .) %>% as.numeric() - 0.5]
	metadata <- metadata %>%
		mutate(
			preterm=ifelse(gacat < 37, 1, 0),
			extreme_preterm=ifelse(gacat < 28, 1, 0),
			sga=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0))
}

# Write out PDF file of results
if (is.null(output_file)) {
	results_pdf <- paste0(input_dir, "performance_curves.pdf")
} else {
	results_pdf <- output_file
}

pdf(results_pdf, height=7, width=7, useDingbats=FALSE)
roc_list <- vector("list", length(neonatal_outcomes))
names(roc_list) <- neonatal_outcomes
auc_list <- vector("character", length(neonatal_outcomes))
names(auc_list) <- neonatal_outcomes

no_overlap_roc_list <- vector("list", length(neonatal_outcomes))
names(no_overlap_roc_list) <- neonatal_outcomes
no_overlap_auc_list <- vector("character", length(neonatal_outcomes))
names(no_overlap_auc_list) <- neonatal_outcomes

# Plot TPR/FPR Curves
for (outcome in neonatal_outcomes) {
	df_for_roc <- merged_results %>%
		filter(!(get(outcome) == 0 & total_conditions >= 1))
	preds_col <- paste0(outcome, "_pred")
	if (isTRUE(rev_preds_direction)) {
		direction_to_use <- ">"
	} else {
		# Typically prediction probabilities where higher values associated with
		# label
		direction_to_use <- "<"
	}

	if (isTRUE(controls_as_positive)) {
		df_for_roc[[outcome]] <- ifelse(df_for_roc[[outcome]] == 0, 1, 0)
	}

	roc_results <- roc(
		response=df_for_roc[[outcome]],
		predictor=df_for_roc[[preds_col]], direction=direction_to_use)

	roc_list[[outcome]] <- roc_results
	auc_label <- paste0("AUC: ", round(roc_results$auc, 3))
	auc_list[[outcome]] <- auc_label
	prediction_pval <- wilcox.test(
		df_for_roc[df_for_roc[[outcome]] == 0, ][[preds_col]],
		df_for_roc[df_for_roc[[outcome]] == 1, ][[preds_col]])
	pval <- prediction_pval$p.value
	roc_plt <- ggroc(
		roc_results, color=outcomes_colors[[outcome]], legacy.axes=TRUE) +
		geom_segment(aes(x=0, xend=1, y=0, yend=1), color="grey", linetype="dashed") +
		xlab("FPR") +
		ylab("TPR") +
		annotate("text", x=0.75, y=0.2, size=10, label=auc_label) +
		ggtitle(paste(outcome, "ROC")) +
		theme(plot.title=element_text(size=20), aspect.ratio=1)
	print(roc_plt)
}

names(roc_list) <- gsub("_any", "", names(roc_list)) %>% toupper()
combined_plt <- ggroc(roc_list, aes="color", legacy.axes=TRUE) +
	scale_color_tableau() +
	geom_segment(aes(x=0, xend=1, y=0, yend=1), color="grey", linetype="dashed") +
	xlab("FPR") +
	ylab("TPR") +
	annotate("text", x=0.5, y=0.3, size=6, color=outcomes_colors[["bpd_any"]],
					 label=paste("BPD", auc_list[["bpd_any"]]), hjust=0) +
	annotate("text", x=0.5, y=0.25, size=6, color=outcomes_colors[["ivh_any"]],
					 label=paste("IVH", auc_list[["ivh_any"]]), hjust=0) +
	annotate("text", x=0.5, y=0.2, size=6, color=outcomes_colors[["nec_any"]],
					 label=paste("NEC", auc_list[["nec_any"]]), hjust=0) +
	annotate("text", x=0.5, y=0.15, size=6, color=outcomes_colors[["rop_any"]],
					 label=paste("ROP", auc_list[["rop_any"]]), hjust=0) +
	ggtitle("ROC") +
	labs(color="Neonatal\nOutcome") +
	theme(aspect.ratio=1)
print(combined_plt)

prcurve_list <- vector("list", length(neonatal_outcomes))
names(prcurve_list) <- neonatal_outcomes
aupr_list <- vector("character", length(neonatal_outcomes))
names(aupr_list) <- neonatal_outcomes

if (isTRUE(add_random_to_prc)) {
	rand_prcurve_list <- vector("list", length(neonatal_outcomes))
	names(rand_prcurve_list) <- neonatal_outcomes
	rand_aupr_list <- vector("character", length(neonatal_outcomes))
	names(rand_aupr_list) <- neonatal_outcomes
}

# Plot Precision-Recall Curves
for (outcome in neonatal_outcomes) {
	df_for_roc <- merged_results %>%
		filter(!(get(outcome) == 0 & total_conditions >= 1))
	preds_col <- paste0(outcome, "_pred")

	if (isTRUE(rev_preds_direction)) {
		df_for_roc <- df_for_roc %>%
			mutate_at(preds_col, sigmoid_transform) %>%
			mutate_at(preds_col, function(x) {1 - x})
	}
	if (isTRUE(controls_as_positive)) {
		event_level_to_use <- "first"
	} else {
		event_level_to_use <- "second"
	}
	pr_df <- pr_curve(
		df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
		!!outcome, !!preds_col, event_level=event_level_to_use) %>%
		rename(Recall=recall, Precision=precision)
	pr_auc_value <- pr_auc(
		df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
		!!outcome, !!preds_col, event_level=event_level_to_use)

	if (isTRUE(add_random_to_prc)) {
		set.seed(101)
		rand_df_for_roc <- mutate(
			df_for_roc, random_classifier=runif(nrow(df_for_roc), 0, 1))
		rand_pr_df <- pr_curve(
			rand_df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, random_classifier, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		rand_auprc_value <- pr_auc(
			rand_df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, random_classifier, event_level=event_level_to_use)
		rand_pr_df[["Neonatal\nOutcome"]] <- paste0(outcome, "\n(Random\nClassifier)")
		rand_prcurve_list[[outcome]] <- rand_pr_df
		rand_aupr_label <- paste0(round(rand_auprc_value$.estimate, 3))
		rand_aupr_list[[outcome]] <- rand_aupr_label
	}

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

	if (isTRUE(reference_points)) {
		get_pr_point <- function(predicted_labels, true_labels, point_name, positive="1") {
			list(
				Recall=caret::sensitivity(predicted_labels %>% factor(), true_labels %>% factor(), positive=positive),
				Precision=posPredValue(predicted_labels %>% factor(), true_labels %>% factor(), positive=positive),
				Name=point_name)
		}
		merged_metadata <- merge(df_for_roc, metadata, by="row_id")
		extreme_preterm_pt <- get_pr_point(
			merged_metadata$extreme_preterm, merged_metadata[[outcome]], point_name="extreme\npreterm")
		sga_pt <- get_pr_point(
			merged_metadata$sga, merged_metadata[[outcome]], point_name="SGA")
		ref_points <- bind_rows(extreme_preterm_pt, sga_pt)
		plt <- plt +
			geom_point(data=ref_points, mapping=aes(x=Recall, y=Precision), size=4) +
			geom_text(data=ref_points, aes(label=Name),
								nudge_x=0.025, nudge_y=0.025, size=6, hjust=0, vjust=0)
	}
	print(plt)
}

combined_plt <- ggplot(bind_rows(prcurve_list), aes(x=Recall, y=Precision, color=`Neonatal\nOutcome`)) +
	geom_path() +
	scale_color_tableau() +
	ylim(0, 1) +
	annotate("text", x=0.5, y=0.22, size=6, color=outcomes_colors[["bpd_any"]],
					 label=paste("BPD", aupr_list[["bpd_any"]]), hjust=0) +
	annotate("text", x=0.5, y=0.15, size=6, color=outcomes_colors[["ivh_any"]],
					 label=paste("IVH", aupr_list[["ivh_any"]]), hjust=0) +
	annotate("text", x=0.5, y=0.08, size=6, color=outcomes_colors[["nec_any"]],
					 label=paste("NEC", aupr_list[["nec_any"]]), hjust=0) +
	annotate("text", x=0.5, y=0.01, size=6, color=outcomes_colors[["rop_any"]],
					 label=paste("ROP", aupr_list[["rop_any"]]), hjust=0) +
	ggtitle("Precision-Recall Curve") +
	labs(color="Neonatal\nOutcome") +
	theme(aspect.ratio=1)
print(combined_plt)

if (isTRUE(add_random_to_prc)) {
	rand_classifier_colors <- seq_gradient_pal(
		"#4d4d4d", "#c4c4c4")(seq(0, 1, length.out=length(neonatal_outcomes)))
	names(rand_classifier_colors) <- paste0(neonatal_outcomes, "\n(Random\nClassifier)")
	expanded_colors <- c(outcomes_colors, rand_classifier_colors)
	rand_linetypes <- c("longdash", "dotted", "dashed", "dotdash")
	outcomes_linetypes <- rep("solid", length(neonatal_outcomes))
	expanded_linetypes <- c(outcomes_linetypes, rand_linetypes)
	names(expanded_linetypes) <- names(expanded_colors)

	create_auprc_label <- function(neonatal_outcome, aupr_list, rand_aupr_list) {
		label <- paste0(
			gsub("_any", "", neonatal_outcome) %>% toupper(),
			" ",
			aupr_list[[neonatal_outcome]],
			" (Baseline: ",
			rand_aupr_list[[neonatal_outcome]],
			")"
		)
	}

	all_curves <- c(prcurve_list, rand_prcurve_list)
	combined_plt <- ggplot(
		bind_rows(prcurve_list), aes(x=Recall, y=Precision,
		color=`Neonatal\nOutcome`, linetype=`Neonatal\nOutcome`)) +
		geom_path() +
		scale_color_manual(values=expanded_colors) +
		scale_linetype_manual(values=expanded_linetypes) +
		ylim(0, 1) +
		annotate("text", x=0.05, y=0.22, size=6, color=outcomes_colors[["bpd_any"]],
						 label=create_auprc_label("bpd_any", aupr_list, rand_aupr_list), hjust=0) +
		annotate("text", x=0.05, y=0.15, size=6, color=outcomes_colors[["ivh_any"]],
						 label=create_auprc_label("ivh_any", aupr_list, rand_aupr_list), hjust=0) +
		annotate("text", x=0.05, y=0.08, size=6, color=outcomes_colors[["nec_any"]],
						 label=create_auprc_label("nec_any", aupr_list, rand_aupr_list), hjust=0) +
		annotate("text", x=0.05, y=0.01, size=6, color=outcomes_colors[["rop_any"]],
						 label=create_auprc_label("rop_any", aupr_list, rand_aupr_list), hjust=0) +
		ggtitle("Precision-Recall Curve") +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1)
	print(combined_plt)
}

if (isTRUE(summary_tables)) {
	#grid.newpage()
	#grid.table(performance_summary)
	#grid.newpage()
	#grid.table(combined_performance_summary)
}
dev.off()

