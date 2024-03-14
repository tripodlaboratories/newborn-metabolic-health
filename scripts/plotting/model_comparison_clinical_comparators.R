# Implement plotting for model comparison and other clinical comparators
library(readr)
library(dplyr)
library(data.table)
library(ggplot2)
library(ggthemes)

# Read in results data from different directories
model_comparison_dir <- "./results/model_comparison/multi_task/"
deep_learning_dir <- "./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_10/"

models_for_comparison <- c("lasso", "en", "rf", "hgbc", "xgboost")

score_dfs <- list()
for (m in models_for_comparison) {
	score_df <- fread(paste0(model_comparison_dir, m, "/scores.csv")) %>%
		mutate(model=m)
	score_dfs <- append(score_dfs, list(score_df))
}

nn_scores <- fread(paste0(deep_learning_dir, "/scores.csv")) %>%
	mutate(model="deep neural network")
score_dfs <- append(score_dfs, list(nn_scores))

all_scores <- bind_rows(score_dfs)
scores_table <- all_scores %>% group_by(task, model) %>%
	summarize(
		mean_auroc=mean(auroc_strict), sd_auroc=sd(auroc_strict),
		mean_aupr=mean(aupr_strict), sd_aupr=sd(aupr_strict))
write_csv(scores_table, "./results/model_comparison/model_comparison_scores.csv")

# Repeat results for clincal comparator models
clinical_comparator_dir <- "./results/clinical_comparators/"
# Feature sets are expected to be directory names
clinical_feature_sets <- c("bwtga", "apgar_only", "minimal_vars", "additional_risk_vars")

all_clinical_comparators <- list()
for (feat_set_dir in clinical_feature_sets) {
	clinical_models <- c("en", "lasso", "lr")
	clin_score_dfs <- list()
	for (cm in clinical_models) {
		clinical_model_filename <- paste0(
			clinical_comparator_dir, feat_set_dir, "/", cm, "/scores.csv")
		score_df <- fread(clinical_model_filename) %>%
			mutate(model=cm)
		clin_score_dfs <- append(clin_score_dfs, list(score_df))
	}
	all_clin_scores <- bind_rows(clin_score_dfs)

	clin_scores_table <- all_clin_scores %>% group_by(task, model) %>%
		summarize(
			mean_auroc=mean(auroc_strict), sd_auroc=sd(auroc_strict),
			mean_aupr=mean(aupr_strict), sd_aupr=sd(aupr_strict)) %>%
	mutate(feature_set=feat_set_dir)
	all_clinical_comparators <- append(
		all_clinical_comparators, list(clin_scores_table))
}

all_clinical_comparators <- append(
	all_clinical_comparators,
	list(nn_scores %>%
			 group_by(task, model) %>%
			 summarize(
				mean_auroc=mean(auroc_strict), sd_auroc=sd(auroc_strict),
				mean_aupr=mean(aupr_strict), sd_aupr=sd(aupr_strict)) %>%
			 mutate(feature_set="nbs_metabolites")))
clinical_comparators_df <- bind_rows(all_clinical_comparators)

write_csv(
	clinical_comparators_df,
	"./results/clinical_comparators/feature_sets_models_scores.csv")

