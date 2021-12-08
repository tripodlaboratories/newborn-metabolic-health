# Compare models that integrate clinical variables across different
# subsets of variables
library(readr)
library(dplyr)
library(data.table)
library(ggplot2)
library(ggthemes)

source("./scripts/plotting/prediction_scoring.R")

# Read in results data from different directories
clinical_results_dir <- "./results/deep_mtl/neonatal_clinical_metabolic/"
deep_learning_dir <- "./results/deep_mtl/neonatal_bottleneck_fulltest/ensemble_bottle_10/"
deep_health_index_dir <- "./results/deep_mtl/neonatal_bottleneck_fulltest/ensemble_bottle_1/"

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

# Feature sets are expected to be directory names
clinical_feature_sets <- c(
	"bwtga", "apgar_only", "minimal_vars", "additional_risk_vars", "clinical_features_only")
comparison_model_name <- "ensemble_bottle_10"

clinical_feature_scores <- list()
for (feat_set_dir in clinical_feature_sets) {
	clinical_model_filename <- paste0(
		clinical_results_dir, feat_set_dir, "/", comparison_model_name,
		"/scores.csv")
	score_df <- fread(clinical_model_filename) %>%
		mutate(feature_set=feat_set_dir, model="Deep Learning") %>%
		group_by(task, model, feature_set) %>%
		summarize(
			mean_auroc=mean(auroc_strict), sd_auroc=sd(auroc_strict),
			mean_aupr=mean(aupr_strict), sd_aupr=sd(aupr_strict),
			.groups="drop")

	clinical_feature_scores <- append(clinical_feature_scores, list(score_df))
}

nn_scores <- fread(paste0(deep_learning_dir, "/scores.csv")) %>%
	mutate(model="deep neural network")
clinical_feature_scores <- append(
	clinical_feature_scores,
	list(nn_scores %>%
			 mutate(feature_set="NBS Metabolites Only") %>%
			 group_by(task, model, feature_set) %>%
			 summarize(
				mean_auroc=mean(auroc_strict), sd_auroc=sd(auroc_strict),
				mean_aupr=mean(aupr_strict), sd_aupr=sd(aupr_strict))))
clinical_feature_scores_df <- bind_rows(clinical_feature_scores)

write_csv(
	clinical_feature_scores_df,
	"./results/neonatal_clinical_metabolic/clinical_feature_set_model_scores.csv")

# Repeat with metabolic health index using predictions
overall_true_vals <- fread("./data/processed/neonatal_conditions.csv") %>%
	select(row_id, one_of(c("bpd_any", "ivh_any", "nec_any", "rop_any"))) %>%
	mutate(total_conditions=select(., -row_id) %>% rowSums()) %>%
	pivot_longer(!any_of(c("row_id", "total_conditions")), names_to="outcome", values_to="true_value")

comparison_model_name <- "ensemble_bottle_1"
clinical_feature_preds <- list()
for (feat_set_dir in clinical_feature_sets) {
	clinical_model_filename <- paste0(
		clinical_results_dir, feat_set_dir, "/", comparison_model_name,
		"/bottleneck.csv")
	preds <- fread(clinical_model_filename) %>%
		rename(row_id=V1, healthy_infant=bottleneck_unit_0) %>%
		mutate(bpd_any=healthy_infant,
					 ivh_any=healthy_infant,
					 nec_any=healthy_infant,
					 rop_any=healthy_infant) %>%
		select(-fold, -epoch, -healthy_infant) %>%
		pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
		mutate(model="Deep Learning")
	pred_df <- merge(preds, overall_true_vals, by=c("row_id", "outcome")) %>%
		mutate(feature_set=feat_set_dir)
	clinical_feature_preds <- append(clinical_feature_preds, list(pred_df))
}

# Add existing neural network results
nn_preds <- fread(paste0(deep_health_index_dir, "/bottleneck.csv")) %>%
	rename(row_id=V1, healthy_infant=bottleneck_unit_0) %>%
	mutate(bpd_any=healthy_infant,
				 ivh_any=healthy_infant,
				 nec_any=healthy_infant,
				 rop_any=healthy_infant) %>%
	select(-fold, -epoch, -healthy_infant) %>%
	pivot_longer(!any_of(c("row_id", "iter")), names_to="outcome", values_to="predicted") %>%
	mutate(model="Deep Learning")
nn_pred_df <- merge(nn_preds, overall_true_vals, by=c("row_id", "outcome")) %>%
	mutate(feature_set="NBS Metabolites Only")
clinical_feature_preds <- append(clinical_feature_preds, list(nn_pred_df))

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
		evaluation_metric == "AUROC" ~ "Metabolic Health Index",
		TRUE ~ "other")) %>%
	select(-evaluation_metric, -dataset) %>%
	merge(
		overall_true_vals %>% select(row_id, outcome, total_conditions) %>% distinct(),
		by=c("row_id", "outcome"), all.x=TRUE, all.y=FALSE) %>%
	mutate(feature_set="NBS Metabolites Only")
clinical_feature_preds <- append(clinical_feature_preds, list(sgdis_hi_df))

# Get scores for health index
clin_feat_preds_df <- bind_rows(clinical_feature_preds)
grouping_vars <- c("outcome", "model", "feature_set", "iter")
scores <- get_scores(
	clin_feat_preds_df, initial_grouping_vars=grouping_vars, event_level="first")

# Reformat the scores
scores <- scores %>%
	mutate(outcome=gsub("_any", "", outcome) %>% toupper())

write_csv(
	scores,
	"./results/neonatal_clinical_metabolic/clinical_feature_set_health_index_scores.csv")

