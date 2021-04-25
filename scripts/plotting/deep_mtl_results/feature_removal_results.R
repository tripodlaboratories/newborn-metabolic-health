# Plot feature removal.
# example output file: ./results/deep_mtl/plots/feature_removal.pdf
library(dplyr)
library(tidyr)
library(readr)
library(data.table)
library(ggplot2)
library(ggthemes)
library(logger)
library(optparse)
log_threshold(DEBUG)

# Set up arguments
option_list <- list(
  make_option(c("-i", "--input"),
              type="character",
              help="Input directory containing feature removal results."),
	make_option(c("-o", "--output_file"),
							type="character",
							help="output PDF file for saving results."),
	make_option(c("--n_features"), type="integer", default=46,
							help="highest number of features"),
	make_option(c("--feature_file"), type="character", default=NULL,
							help="File containing the full set of features to infer feature order."),
	make_option(c("--total_aupr_ranks"), action="store_true", default=FALSE,
							help="Specify if results are based on total aupr ranks, otherwise results are based on outcome ranks."),
	make_option(c("--interactive_debug"), action="store_true", default=FALSE,
							help="Sets testing options for interactive debugging")
)
opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input
output_file <- opts$output_file
highest_num_features <- opts$n_features
feature_file <- opts$feature_file
total_aupr_ranks <- opts$total_aupr_ranks
interactive_debug <- opts$interactive_debug

if (isTRUE(interactive_debug)) {
  input_dir <- "./results/deep_mtl/neonatal/feature_removal/"
	output_file <- NULL
	feature_file <- "./results/deep_mtl/neonatal/feature_removal/ensemble_46_feat_total_aupr_ranks/features.txt"
	total_aupr_ranks <- TRUE
}

# Read in covariate data
outcomes <- read_lines("./config/neonatal_covariates.txt") %>% sort()
all_results <- Sys.glob(paste0(input_dir, "*"))

get_removal_scores <- function(outcome_dirs, model_name) {
	model_dirs <- outcome_dirs[grepl(model_name, outcome_dirs)]
	score_files <- paste0(model_dirs, "/scores.csv")
	scores <- rbindlist(lapply(score_files, fread))
	mean_scores <- scores[,
		lapply(.SD, mean), by=c("task", "n_features"),
		.SDcols=c("pval", "auroc", "aupr", "auroc_strict", "aupr_strict")]
}

# Iterate over outcomes and plot results
model_names <- c("ensemble", "ensemble_bottleneck")
feature_aupr_ranks <- read_csv(
	"./results/neonatal/metabolite_scores/neonatal_metabolite_aupr_no_overlap.csv")

theme_set(theme_base(base_size=18))
outcomes_colors <- tableau_color_pal("Tableau 10")(length(outcomes))
names(outcomes_colors) <- outcomes %>% gsub("_any", "", .) %>% toupper()

if (isTRUE(total_aupr_ranks)) {
	outcomes_for_plots <- c("total_aupr_ranks")
} else {
  outcomes_for_plots <- outcomes
}
pdf(output_file, height=7, width=10.5, useDingbats=FALSE)
for (outcome in outcomes_for_plots) {
	outcome_dirs <- all_results[grepl(outcome, all_results)]

	if (is.null(feature_file)) {
		log_warn("Inferring feature ranks from outcome-based ranks, this may lead to misleading X axis")
		outcome_feature_ranks <- feature_aupr_ranks %>% arrange(get(outcome))
		feature_df <- outcome_feature_ranks %>%
			rename(raw_feature_name=metabolite) %>%
			mutate(n_features=rev(seq(dim(outcome_feature_ranks)[1])))
		metabolite_labels <- fread('./config/metabolite_labels.csv') %>%
			select(raw_feature_name, metabolite=feature_label)
		feature_df <- merge(feature_df, metabolite_labels, by="raw_feature_name")
		feature_df <- feature_df[match(outcome_feature_ranks$metabolite, feature_df$raw_feature_name), ] %>%
			mutate(metabolite=factor(metabolite, levels=metabolite))
	} else {
		features_from_file <- read_lines(feature_file)
		feature_df <- data.frame(
			raw_feature_name=rev(features_from_file),
			n_features=rev(seq(length(features_from_file))))
		metabolite_labels <- fread('./config/metabolite_labels.csv') %>%
			select(raw_feature_name, metabolite=feature_label)
		feature_df <- merge(feature_df, metabolite_labels, by="raw_feature_name")
		feature_df <- feature_df[match(rev(features_from_file), feature_df$raw_feature_name), ] %>%
			mutate(metabolite=factor(metabolite, levels=metabolite))
	}

	for (model_name in model_names) {
		scores <- get_removal_scores(outcome_dirs, model_name)
		# Create a total AUPR columns from AUPR scores
		scores <- scores %>% select(task, n_features, aupr_strict) %>%
			spread(task, aupr_strict) %>%
			mutate(`Total\nAUPR`=rowSums(.[, ..outcomes])) %>%
			gather("task", "aupr_strict", -n_features)

		# Set up labels and plot
		scores <- merge(scores, feature_df, by="n_features") %>%
			mutate(task:=ifelse(task=="Total\nAUPR", task, gsub("_any", "", task) %>% toupper()))
		outcomes_colors <- c(outcomes_colors, c(`Total\nAUPR`="black"))
		plt <- ggplot(
			scores %>% filter(task != "Total\nAUPR"),
			aes(x=metabolite, y=aupr_strict, color=task, group=task)) +
			geom_line() +
			geom_point() +
			scale_color_manual(values=outcomes_colors) +
			theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1, size=11)) +
			ylab("AUPR") +
			ggtitle(paste("feature ranks:", outcome, ", model:", model_name))
		print(plt)

		plt <- ggplot(
			scores %>% filter(task == "Total\nAUPR"),
			aes(x=metabolite, y=aupr_strict, color=task, group=task)) +
			geom_line() +
			geom_point() +
			scale_color_manual(values=outcomes_colors) +
			theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1, size=11)) +
			ylab("AUPR") +
			ggtitle(paste("feature ranks:", outcome, ", model:", model_name))
		print(plt)
	}

	if (isTRUE(total_aupr_ranks)) {
		feature_aupr_ranks <- feature_aupr_ranks %>%
			rename(raw_feature_name=metabolite) %>%
			mutate(total_aupr=rowSums(.[outcomes]))
		feature_aupr_df <- merge(feature_aupr_ranks, metabolite_labels, by="raw_feature_name") %>%
			arrange(total_aupr) %>%
			mutate(metabolite=factor(metabolite, levels=metabolite))
		plt <- ggplot(feature_aupr_df, aes(x=metabolite, y=total_aupr, group=1)) +
			geom_col() +
			theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1, size=11)) +
			ylab("Total AUPR") +
			ggtitle("Feature Importance Ranks Across All Outcomes")
		print(plt)
	}
}
dev.off()

