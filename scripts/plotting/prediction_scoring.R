# Helper functions for scoring predictions from multiple model sources
library(readr)
library(tidyr)
library(dplyr)
library(data.table)
library(yardstick)

# Helper functions
get_scores <- function(
	df,
	true_val_col="true_value",
	pred_col="predicted",
	total_conditions_col="total_conditions",
	event_level="second",
	strict_negative=TRUE,
	initial_grouping_vars=c("outcome", "model", "iter"),
	iter_col="iter") {
	###
	###
	df %>%
		group_by(!!!syms(initial_grouping_vars)) %>%
		summarize(
			auroc=create_score_col(
				roc_auc_vec, get(true_val_col), get(pred_col), get(total_conditions_col),
				event_level, strict_negative),
			aupr=create_score_col(
				pr_auc_vec, get(true_val_col), get(pred_col), get(total_conditions_col),
				event_level, strict_negative),
			.groups="drop") %>%
		group_by(!!!syms(
			initial_grouping_vars[initial_grouping_vars != iter_col])) %>%
		summarize(
			mean_auroc=mean(auroc),
			sd_auroc=sd(auroc),
			mean_aupr=mean(aupr),
			sd_aupr=sd(aupr),
			.groups="drop")
}

create_score_col <- function(
	score_function, true_value, predicted, total_conditions,
	event_level="second", strict_negative=TRUE) {
	if (isTRUE(strict_negative)) {
		negatives_with_other_conditions <- (true_value == 0) & (total_conditions > 0)
		predicted <- predicted[!negatives_with_other_conditions]
		true_value <- true_value[!negatives_with_other_conditions]
	}
	score_function(true_value %>% factor(), predicted, event_level=event_level)
}

create_performance_curve <- function(
	df, curve_function,
	true_value_col="true_value",
	predicted_col="predicted",
	total_conditions_col="total_conditions",
	event_level="second", strict_negative=TRUE, annotation_cols=NULL) {
	# args:
	#    curve_function: pr_curve or roc_curve from yardstick
	#    annotation_cols: list of columns to annotate curve dataframe with, should be
	#        only one value in the original dataframe
	#
	if (isTRUE(strict_negative)) {
		negatives_with_other_conditions <- (
			df[[true_value_col]] == 0) & (df[[total_conditions_col]] > 0)
		df <- df[!negatives_with_other_conditions, ]
	}
	curve_df <- curve_function(
		df, get(true_value_col) %>% factor(), !!predicted_col,
		event_level=event_level)

	if (!is.null(annotation_cols)) {
		for (ac in annotation_cols) {
			curve_df[[ac]] <- df[[ac]] %>% unique()
		}
	}
	return(curve_df)
}

