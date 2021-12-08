# Analysis script using predictions directly for clinical comparators
library(readr)
library(tidyr)
library(dplyr)
library(data.table)
library(yardstick)
library(ggplot2)
library(ggthemes)

source("./scripts/plotting/prediction_scoring.R")

# Clinical Comparator Models
clinical_comparator_dir <- "./results/clinical_comparators/"
comparator_health_index_dir <- "./results/clinical_comparators/as_health_index/"

# Feature sets are expected to be directory names
clinical_feature_sets <- c("bwtga", "apgar_only", "minimal_vars", "additional_risk_vars")
clinical_models <- c("en", "lasso", "lr")
clinical_model_rename_map <- c("EN", "Lasso", "Logistic Regression")
names(clinical_model_rename_map) <- clinical_models

# Add the subgroup discovery predictions
# There are two different subgroup discovery experiments, one on model
# predictions and one on the health index
subgroup_discovery_dir <- "./results/subgroup_discovery/"
sgdis_model_preds <- paste0(
	subgroup_discovery_dir,
	"model_predictions/top_k_subgroup_preds_over_iters.csv")
sgdis_health_index <- paste0(
	subgroup_discovery_dir,
	"metabolic_health_index/top_k_subgroup_preds_over_iters.csv")

overall_true_vals <- fread("./data/processed/neonatal_conditions.csv") %>%
	select(row_id, one_of(c("bpd_any", "ivh_any", "nec_any", "rop_any"))) %>%
	mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
	pivot_longer(!any_of(c("row_id", "total_conditions")), names_to="outcome", values_to="true_value")

# Within each clinical variable feature set there are linear models that use
# that feature set to predic the adverse outcomes of prematurity
all_clinical_comparators <- list()
for (feat_set_dir in clinical_feature_sets) {
	clin_pred_dfs <- list()
	for (cm in clinical_models) {
		clinical_model_dir <- paste0(
			clinical_comparator_dir, feat_set_dir, "/", cm, "/")
		preds <- fread(paste0(clinical_model_dir, "preds.csv.gz")) %>%
			select(-fold) %>%
			pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
			mutate(model=clinical_model_rename_map[[cm]])
		true_vals <- fread(paste0(clinical_model_dir, "true_vals.csv")) %>%
			filter(iter == 0) %>%
			select(-fold, -iter) %>%
			mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
			pivot_longer(!any_of(c("row_id", "total_conditions")), names_to="outcome", values_to="true_value")
		pred_df <- merge(preds, true_vals, by=c("row_id", "outcome"))

		clin_pred_dfs <- append(clin_pred_dfs, list(pred_df))
	}

	all_clin_preds <- bind_rows(clin_pred_dfs) %>%
		mutate(feature_set=feat_set_dir)
	all_clinical_comparators <- append(
		all_clinical_comparators, list(all_clin_preds))
}

# Append neural network predictions
deep_learning_dir <- "./results/deep_mtl/neonatal/fulltest/ensemble/"
nn_preds <- fread(paste0(deep_learning_dir, "preds.csv.gz")) %>%
	rename(row_id=V1) %>%
	select(-fold, -epoch) %>%
	pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
	mutate(model="Neural Network")
nn_true_vals <- fread(paste0(deep_learning_dir, "/true_vals.csv")) %>%
		filter(iter == 0) %>%
		select(-fold, -iter) %>%
		mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
		pivot_longer(!any_of(c("row_id", "outcome", "total_conditions")), names_to="outcome", values_to="true_value")
nn_pred_df <- merge(nn_preds, nn_true_vals, by=c("row_id", "outcome")) %>%
	mutate(feature_set="NBS Metabolites")
all_clinical_comparators <- append(all_clinical_comparators, list(nn_pred_df))

# Append subgroup discovery results, and match colnames and expected
# values, subgroup discovery results also need to be split by the search method
sgdis_preds_df <- fread(sgdis_model_preds) %>%
	rename(predicted=preds, true_value=true_vals) %>%
	filter(evaluation_metric != "AVG Precision") %>%
	mutate(model=case_when(
		evaluation_metric == "AUROC" ~ "Subgroup-Focused Metabolic Health Index",
		TRUE ~ "other")) %>%
	select(-evaluation_metric, -dataset) %>%
	merge(
		overall_true_vals %>% select(row_id, outcome, total_conditions) %>% distinct(),
		by=c("row_id", "outcome"), all.x=TRUE, all.y=FALSE) %>%
	mutate(feature_set="NBS Metabolites")
all_clinical_comparators <- append(all_clinical_comparators, list(sgdis_preds_df))

# Score the clinical comparators for model predictions
comparator_df <- bind_rows(all_clinical_comparators)
grouping_vars <- c("outcome", "model", "feature_set", "iter")
scores <- get_scores(comparator_df, initial_grouping_vars=grouping_vars)
write_csv(scores, "./results/clinical_comparators/comparator_model_scores.csv")

# Score the clinical comparators when used as a metabolic health index
overall_true_vals <- fread("./data/processed/neonatal_conditions.csv") %>%
	select(row_id, one_of(c("bpd_any", "ivh_any", "nec_any", "rop_any"))) %>%
	mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
	pivot_longer(!any_of(c("row_id", "total_conditions")), names_to="outcome", values_to="true_value")

all_clinical_comparators <- list()
for (feat_set_dir in clinical_feature_sets) {
	clin_pred_dfs <- list()
	for (cm in clinical_models) {
		clinical_model_dir <- paste0(
			comparator_health_index_dir, feat_set_dir, "/", cm, "/")
		preds <- fread(paste0(clinical_model_dir, "preds.csv.gz")) %>%
			select(-fold) %>%
			mutate(
				bpd_any=healthy_infant,
				ivh_any=healthy_infant,
				nec_any=healthy_infant,
				rop_any=healthy_infant) %>%
		select(-healthy_infant) %>%
		pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
			mutate(model=clinical_model_rename_map[[cm]])

		pred_df <- merge(preds, overall_true_vals, by=c("row_id", "outcome"))
		clin_pred_dfs <- append(clin_pred_dfs, list(pred_df))
	}

	all_clin_preds <- bind_rows(clin_pred_dfs) %>%
		mutate(feature_set=feat_set_dir)
	all_clinical_comparators <- append(
		all_clinical_comparators, list(all_clin_preds))
}

# Read in deep neural network metabolic health index values
deep_health_index_dir <- "./results/deep_mtl/neonatal_bottleneck_fulltest/ensemble_bottle_1/"
nn_preds <- fread(paste0(deep_health_index_dir, "/bottleneck.csv")) %>%
	rename(row_id=V1, healthy_infant=bottleneck_unit_0) %>%
	mutate(bpd_any=healthy_infant,
				 ivh_any=healthy_infant,
				 nec_any=healthy_infant,
				 rop_any=healthy_infant) %>%
	select(-fold, -epoch, -healthy_infant) %>%
	pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
	mutate(model="Neural Network")
nn_pred_df <- merge(nn_preds, overall_true_vals, by=c("row_id", "outcome")) %>%
	mutate(feature_set="NBS Metabolites")
all_clinical_comparators <- append(all_clinical_comparators, list(nn_pred_df))

# Read in subgroup discovery results using the metabolic health index
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
		evaluation_metric == "AUROC" ~ "Subgroup-Focused Metabolic Health Index",
		TRUE ~ "other")) %>%
	select(-evaluation_metric, -dataset) %>%
	merge(
		overall_true_vals %>% select(row_id, outcome, total_conditions) %>% distinct(),
		by=c("row_id", "outcome"), all.x=TRUE, all.y=FALSE) %>%
	mutate(feature_set="NBS Metabolites")
all_clinical_comparators <- append(all_clinical_comparators, list(sgdis_hi_df))

# Score predictions
comparator_df <- bind_rows(all_clinical_comparators)
scores <- get_scores(
	comparator_df, initial_grouping_vars=grouping_vars, event_level="first")

write_csv(
	scores,
	"./results/clinical_comparators/comparator_health_index_scores.csv")

