# Plot gestational age results
library(readr)
library(data.table)
library(dplyr)
library(tidyr)
library(ggthemes)
library(ggplot2)

# Read in neonatal data
neonatal <- read_csv("./data/processed/neonatal_conditions.csv", col_types=cols())
outcomes <- read_lines("./config/neonatal_covariates.txt")
metadata <- read_csv("./data/processed/metadata.csv", col_types=cols())

neonatal_outcomes <- neonatal %>%
	select(one_of(c("row_id", outcomes))) %>%
	mutate(num_conditions=rowSums(.[outcomes])) %>%
	mutate(neonatal_outcome=case_when(
		bpd_any == 1 & num_conditions == 1 ~ "BPD",
		rop_any == 1 & num_conditions == 1 ~ "ROP",
		ivh_any == 1 & num_conditions == 1 ~ "IVH",
		nec_any == 1 & num_conditions == 1 ~ "NEC",
		num_conditions >= 1 ~ "Overlapping",
		num_conditions == 0 ~ "None",
		TRUE ~ "None"))
neonatal_outcomes <- merge(
	neonatal_outcomes,
	metadata %>% select(one_of(c("row_id", "_dthind"))), by="row_id") %>%
	rename(mortality=`_dthind`) %>%
  mutate(mortality=case_when(
    mortality == 1 ~ "Neonatal\nDeath",
    mortality == 0 ~ "Survived",
    mortality == 2 ~ "Postneonatal\nDeath",
    TRUE ~ "Unknown"))

# Read in results
results_dir <- "./results/deep_mtl/gest_age/"
preds_dirs <- list.dirs(results_dir, recursive=FALSE)

theme_set(theme_base(base_size=18))
neonatal_condition_colors <- c(
  "BPD"="#4E79A7", "IVH"="#F28E2B", "NEC"="#E15759", "ROP"="#76B7B2",
  "Overlapping"="#027B8E", "None"="light grey")
mortality_colors <- c("Neonatal\nDeath"="#d72c16", "Survived"="light grey",
                      "Postneonatal\nDeath"="#a10115", "Unknown"="#f1f3ce")
outlier_colors <- c("outlier"="#4d5b8c", "non-outlier"="light grey")

# Function to label samples that are outliers in terms of predictions -
# Possibly standard deviation away from mean prediction for each gestational age
# category
# Currently designing for group_by gestational age, apply to column
label_outliers <- function(predictions, sd_threshold=1) {
	preds_mean <- mean(predictions)
	preds_sd <- sd(predictions)
	outliers <- ifelse(
		abs(predictions - preds_mean) > preds_sd * sd_threshold,
		"outlier", "non-outlier")
}

# Write out plots
pdf("./results/deep_mtl/gest_age/predictions.pdf", useDingbats=FALSE)
for (preds_dir in preds_dirs) {
	model_name <- gsub(".*/([a-z|_]*)$", "\\1", preds_dir)
	preds <- read_csv(paste0(preds_dir, "/preds.csv.gz")) %>%
		rename(gacat_pred=gacat)
	if ("X1" %in% colnames(preds)) preds <- rename(preds, row_id=X1)
	mean_preds <- preds %>% group_by(row_id) %>%
		summarize(gacat_pred=mean(gacat_pred))
	true_vals <- read_csv(paste0(preds_dir, "/true_vals.csv")) %>%
		filter(iter == 0) %>% select(-fold, -iter)
	if ("X1" %in% colnames(true_vals)) true_vals <- rename(true_vals, row_id=X1)
	results_df <- merge(mean_preds, true_vals, by="row_id")
	results_df <- merge(
		results_df, neonatal_outcomes, by="row_id", all.x=TRUE, all.y=FALSE)

	# Add outlier annotation
	results_df <- results_df %>% group_by(gacat) %>%
		mutate(outlier=label_outliers(gacat_pred, sd_threshold=2))
	positional_df <- data.frame(
		min_ga=min(results_df$gacat), max_ga=max(results_df$gacat))

	plt <- ggplot(results_df, aes(x=gacat, y=gacat_pred, color=outlier)) +
		geom_segment(aes(x=min_ga, xend=max_ga, y=min_ga, yend=max_ga), color="grey", linetype=2, data=positional_df) +
		geom_jitter(alpha=0.6, width=0.5) +
		scale_color_manual(values=outlier_colors) +
		ggtitle(paste(model_name, "predictions"))
	print(plt)
	plt <- ggplot(results_df, aes(x=gacat, y=gacat_pred, color=neonatal_outcome)) +
		geom_segment(aes(x=min_ga, xend=max_ga, y=min_ga, yend=max_ga), color="grey", linetype=2, data=positional_df) +
		geom_jitter(alpha=0.6, width=0.5) +
		scale_color_manual(values=neonatal_condition_colors) +
		ggtitle(paste(model_name, "predictions"))
	print(plt)
	plt <- ggplot(results_df, aes(x=gacat, y=gacat_pred, color=mortality)) +
		geom_segment(aes(x=min_ga, xend=max_ga, y=min_ga, yend=max_ga), color="grey", linetype=2, data=positional_df) +
		geom_jitter(alpha=0.6, width=0.5) +
		scale_color_manual(values=mortality_colors) +
		ggtitle(paste(model_name, "predictions"))
	print(plt)

	# Create barplot for outliers and non outliers
	plt <- ggplot(results_df, aes(x=outlier)) +
		geom_bar(aes(fill=mortality)) +
		scale_fill_manual(values=mortality_colors) +
		ggtitle(paste(model_name, "outlier mortality"))
	print(plt)

	results_df <- results_df %>%
		mutate(ga_threshold=ifelse(gacat <= 32, "ga <= 32", "ga > 32"))
	plt <- ggplot(results_df, aes(x=outlier)) +
		geom_bar(aes(fill=mortality)) +
		facet_grid(cols=vars(ga_threshold)) +
		scale_fill_manual(values=mortality_colors) +
		ggtitle(paste(model_name, "outlier mortality"))
	print(plt)

	mortality_percentages <- results_df %>%
		filter(gacat <= 32) %>%
		drop_na(outlier) %>%
		group_by(outlier, mortality) %>%
		count() %>%
		group_by(outlier) %>%
		mutate(percentage=n/sum(n))

	plt <- ggplot(mortality_percentages, aes(x=mortality, y=percentage, fill=mortality)) +
    geom_bar(position="dodge", stat="identity") +
    geom_text(aes(label=scales::percent(percentage),
                  y=percentage), stat="identity", vjust=-.5) +
    labs(y="Percent", fill="mortality") +
    facet_grid(~outlier) +
    scale_y_continuous(labels=scales::percent) +
    scale_fill_manual(values=mortality_colors) +
    theme(legend.position="none", axis.text.x=element_text(size=8)) +
		ggtitle(paste(model_name, "outlier differences at GA < 32"))
	print(plt)

	plt <- ggplot(results_df, aes(x=outlier)) +
		geom_bar(aes(fill=neonatal_outcome)) +
		facet_grid(cols=vars(ga_threshold)) +
		scale_fill_manual(values=neonatal_condition_colors) +
		ggtitle(paste(model_name, "outlier outcomes"))
	print(plt)

	outcome_percentages <- results_df %>%
		filter(gacat <= 32) %>%
		drop_na(outlier) %>%
		group_by(outlier, neonatal_outcome) %>%
		count() %>%
		group_by(outlier) %>%
		mutate(percentage=n/sum(n))

	plt <- ggplot(outcome_percentages, aes(x=neonatal_outcome, y=percentage, fill=neonatal_outcome)) +
    geom_bar(position="dodge", stat="identity") +
    geom_text(aes(label=scales::percent(percentage),
                  y=percentage), stat="identity", vjust=-.5) +
    labs(y="Percent", fill="Neonatal Outcome") +
    facet_grid(~outlier) +
    scale_y_continuous(labels=scales::percent) +
    scale_fill_manual(values=neonatal_condition_colors) +
    theme(legend.position="none", axis.text.x=element_text(size=8)) +
		ggtitle(paste(model_name, "outlier differences at GA < 32"))
	print(plt)
}
dev.off()
