# Analysis script using predictions directly
library(readr)
library(tidyr)
library(dplyr)
library(data.table)
library(yardstick)
library(RColorBrewer)
library(ggplot2)
library(ggthemes)

# Helper functions
source("./scripts/plotting/prediction_scoring.R")

# Read in results data from different directories
model_comparison_dir <- "./results/model_comparison/"
health_index_dir <- "./results/model_comparison/as_health_index/"
deep_learning_dir <- "./results/deep_mtl/neonatal/validation/ensemble/"

# Add the subgroup discovery predictions to incorporate
# There are two different subgroup discovery experiments, one on model
# predictions and one on the health index
subgroup_discovery_dir <- "./results/subgroup_discovery/"
sgdis_model_preds <- paste0(
	subgroup_discovery_dir, "model_predictions/top_k_subgroup_predictions.csv")
sgdis_health_index <- paste0(
	subgroup_discovery_dir, "metabolic_health_index/top_k_subgroup_predictions.csv")

# Begin comparison to models
models_for_comparison <- c("lasso", "en", "rf", "hgbc", "xgboost")
model_label_map <- c("Lasso", "EN", "RF", "HGBC", "XGBoost")
names(model_label_map) <- models_for_comparison

# Use a data structure that stores the individual ROC and
# PR curves, we are already extracting predictions and calculating AUROC and
# AUPRC so it would be good to store the curves themselves somewhere and print
pred_dfs <- list()
for (m in models_for_comparison) {
	preds <- fread(paste0(model_comparison_dir, m, "/preds.csv.gz")) %>%
		select(-fold) %>%
		pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
		mutate(model=model_label_map[[m]])

	# Total conditions tracking is important for scoring by removing cases where
	# the true value is negative but there are other conditions as well
	true_vals <- fread(paste0(model_comparison_dir, m, "/true_vals.csv")) %>%
		filter(iter == 0) %>%
		select(-fold, -iter) %>%
		mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
		pivot_longer(!any_of(c("row_id", "total_conditions")), names_to="outcome", values_to="true_value")

	pred_df <- merge(preds, true_vals, by=c("row_id", "outcome"))
	pred_dfs <- append(pred_dfs, list(pred_df))
}

nn_preds <- fread(paste0(deep_learning_dir, "/preds.csv.gz")) %>%
	rename(row_id=V1) %>%
	select(-fold, -epoch) %>%
	pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
	mutate(model="Neural Network")
nn_true_vals <- fread(paste0(model_comparison_dir, m, "/true_vals.csv")) %>%
		filter(iter == 0) %>%
		select(-fold, -iter) %>%
		mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
		pivot_longer(!any_of(c("row_id", "outcome", "total_conditions")), names_to="outcome", values_to="true_value")
nn_pred_df <- merge(nn_preds, nn_true_vals, by=c("row_id", "outcome"))
pred_dfs <- append(pred_dfs, list(nn_pred_df))

# Append a random predictor
set.seed(101)
random_pred_df <- pred_dfs[[1]]
n_prediction_rows <- length(pred_dfs[[1]][["predicted"]])
random_pred_df[["predicted"]] <- runif(n_prediction_rows)
random_pred_df[["model"]] <- "random predictions"
pred_dfs <- append(pred_dfs, list(random_pred_df))

all_preds <- bind_rows(pred_dfs)

# Append subgroup discovery results, and match colnames and expected
# values, subgroup discovery results also need to be split by the search method
sgdis_preds_df <- fread(sgdis_model_preds) %>%
	rename(predicted=preds, true_value=true_vals) %>%
	filter(evaluation_metric != "AVG Precision") %>%
	# Use only the AUROC scoring metric for comparison here
	mutate(model=case_when(
		evaluation_metric == "AUROC" ~ "Metabolic Health Index",
		TRUE ~ "other")) %>%
	select(-evaluation_metric, -dataset) %>%
	merge(
		all_preds %>% select(row_id, outcome, total_conditions) %>% distinct(),
		by=c("row_id", "outcome"), all.x=TRUE, all.y=FALSE)

# Write out the predictions dataframe
scores <- get_scores(all_preds)
#write_csv(scores, "./results/model_comparison/model_comparison_prediction_scores.csv")

# Create ROC/PR curves using mean preds
all_preds_mean <- all_preds %>% group_by(row_id, outcome, model) %>%
	mutate(predicted=mean(predicted)) %>%
	select(-iter) %>%
	distinct() %>%
	ungroup()
# Subgroup Discovery results already use mean predictions
all_preds_mean <- bind_rows(all_preds_mean, sgdis_preds_df)

all_roc <- all_preds_mean %>% group_by(outcome, model) %>% group_split() %>%
	lapply(create_performance_curve, curve_function=roc_curve,
				 annotation_cols=c("outcome", "model")) %>%
	bind_rows()

all_pr <- all_preds_mean %>% group_by(outcome, model) %>% group_split() %>%
	lapply(create_performance_curve, curve_function=pr_curve,
				 annotation_cols=c("outcome", "model")) %>%
	bind_rows()

# Set up plotting parameters
theme_set(theme_base(base_size=18))
outcomes <- all_preds_mean$outcome %>% unique() %>% sort()
model_colors <- c(
	"EN"="#d17aa5",
	"RF"="#ea715d",
	"HGBC"="#efd89b",
	"Lasso"="#ffb14e",
	"XGBoost"="#66ae9e",
	"Deep Learning"="#a4d9fe",
	"Metabolic Health Index"="#804b90",
	"Random Predictions on Full Dataset"="dark grey")

# Repeate for health index setup
pred_dfs <- list()
overall_true_vals <- fread("./data/processed/neonatal_conditions.csv") %>%
	select(row_id, one_of(c("bpd_any", "ivh_any", "nec_any", "rop_any"))) %>%
	mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
	pivot_longer(!any_of(c("row_id", "total_conditions")), names_to="outcome", values_to="true_value")

for (m in models_for_comparison) {
	preds <- fread(paste0(health_index_dir, m, "/preds.csv.gz")) %>%
		select(-fold) %>%
		mutate(
			bpd_any=healthy_infant,
			ivh_any=healthy_infant,
			nec_any=healthy_infant,
			rop_any=healthy_infant) %>%
		select(-healthy_infant) %>%
		pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
		mutate(model=model_label_map[[m]])

	pred_df <- merge(preds, overall_true_vals, by=c("row_id", "outcome"))
	pred_dfs <- append(pred_dfs, list(pred_df))
}

random_pred_df <- pred_dfs[[1]]
n_prediction_rows <- length(pred_dfs[[1]][["predicted"]])
random_pred_df[["predicted"]] <- runif(n_prediction_rows)
random_pred_df[["model"]] <- "Random Predictions on Full Dataset"
pred_dfs <- append(pred_dfs, list(random_pred_df))

# Need to score NN results separately, since the bottleneck is value is used as
# a prediction
# This is a test directory to compare on the full test set - remove one of
# these directories for the final analysis
deep_health_index_dir <- "./results/deep_mtl/neonatal_bottleneck_fulltest/ensemble_bottle_1/"
nn_preds <- fread(paste0(deep_health_index_dir, "/bottleneck.csv")) %>%
	rename(row_id=V1, healthy_infant=bottleneck_unit_0) %>%
	mutate(
			bpd_any=healthy_infant,
			ivh_any=healthy_infant,
			nec_any=healthy_infant,
			rop_any=healthy_infant) %>%
	select(-fold, -epoch, -healthy_infant) %>%
	pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
	mutate(model="Deep Learning")

nn_pred_df <- merge(nn_preds, overall_true_vals, by=c("row_id", "outcome"))
nn_scores <- get_scores(nn_pred_df, event_level="first")
pred_dfs <- append(pred_dfs, list(nn_pred_df))

all_preds <- bind_rows(pred_dfs)

# Write out the prediction dataframe
health_index_scores <- get_scores(all_preds, event_level="first")
write_csv(health_index_scores, "./results/model_comparison/health_index_prediction_scores.csv")

# Get subgroup discovery scores for the bottleneck index
# Switch the true values back to 0, 1 for healthy infants, subgroup discovery
# operates by treating the healthy infants originally encoded as 0, as a 1 (true
# label
sgdis_hi_df <- fread(sgdis_health_index) %>%
	mutate(true_vals=case_when(
		true_vals == 0 ~ 1,
		true_vals == 1 ~ 0,
		TRUE ~ NA_real_)) %>%
	rename(predicted=preds, true_value=true_vals) %>%
	filter(evaluation_metric != "AVG Precision") %>%
	mutate(model=case_when(
		evaluation_metric == "AUROC" ~ "Metabolic Health Index",
		TRUE ~ "other")) %>%
	select(-evaluation_metric, -dataset) %>%
	merge(
		all_preds %>% select(row_id, outcome, total_conditions) %>% distinct(),
		by=c("row_id", "outcome"), all.x=TRUE, all.y=FALSE)

all_preds_mean <- all_preds %>% group_by(row_id, outcome, model) %>%
	mutate(predicted=mean(predicted)) %>%
	select(-iter) %>%
	distinct() %>%
	ungroup()

# Subgroup Discovery results already use mean predictions
all_preds_mean <- bind_rows(all_preds_mean, sgdis_hi_df)
all_roc <- all_preds_mean %>% group_by(outcome, model) %>% group_split() %>%
	lapply(create_performance_curve, curve_function=roc_curve,
				 annotation_cols=c("outcome", "model"), event_level="first") %>%
	bind_rows()
all_pr <- all_preds_mean %>% group_by(outcome, model) %>% group_split() %>%
	lapply(create_performance_curve, curve_function=pr_curve,
				 annotation_cols=c("outcome", "model"), event_level="first") %>%
	bind_rows()

# Set up plotting parameters

# Plot AUROC and AUPRC curves
pdf("./results/model_comparison/health_index_performance_curves.pdf",
		height=7, width=9, useDingbats=FALSE)
for (o in outcomes) {
	outcome_roc_curve <- all_roc %>% filter(outcome == o) %>%
		mutate(FPR=1-specificity, TPR=sensitivity)
	plt <- ggplot(outcome_roc_curve, aes(x=FPR, y=TPR, color=model)) +
		geom_path() +
		scale_color_manual(values=model_colors) +
		ylim(0, 1) +
		geom_abline(color="grey", linetype="dashed") +
		ylab("True Positive Rate") +
		xlab("False Positive Rate") +
		ggtitle(o) +
		theme(aspect.ratio=1)
	print(plt)

	outcome_pr_curve <- all_pr %>% filter(outcome == o) %>%
		rename(Precision=precision, Recall=recall)
	plt <- ggplot(outcome_pr_curve, aes(x=Recall, y=Precision, color=model)) +
		geom_path() +
		scale_color_manual(values=model_colors) +
		ylim(0, 1) +
		ggtitle(o) +
		theme(aspect.ratio=1)
	print(plt)
}
dev.off()

