# Determine the gestational age distributions of outcomes in the Mednax dataset
library(data.table)
library(dplyr)
library(tidyr)
library(yardstick)
library(ggplot2)
library(ggthemes)

# Read in metadata and outcomes
mednax_dir <- "./external_validation/mednax/processed/"
meta <- fread(paste0(mednax_dir, "mednax_demo_meta_other.csv"))
outcomes <- fread(paste0(mednax_dir, "mednax_outcomes.csv"))

# Set up the gestational age columns and other meta columns of interest
id_col <- "QuestionsRCode"
meta_cols <- c("EGAbest", "Bwt")
ga_col <- "EGAbest"
outcomes_cols <- c("nec_any", "rop_any", "bpd_any", "ivh_any")

# Set up plotting df
meta_select_cols <- c(id_col, meta_cols)
outcomes <- outcomes %>%
	mutate(across(all_of(outcomes_cols), as.factor))
meta_outcomes <- merge(
	meta[, ..meta_select_cols], outcomes, by="QuestionsRCode")

#Write out plots
out_file <- c("./results/external_validation/mednax/outcome_distribution.pdf")
theme_set(theme_base(base_size=18))

majority_label <- function(binary_column) {
	outcome_counts <- table(binary_column)
	first_level_counts <- outcome_counts[1]
	second_level_counts <- outcome_counts[2]

	if (first_level_counts >= second_level_counts) {
		return(names(outcome_counts)[1])
	} else {
		return(names(outcome_counts)[2])
	}
}

pdf(out_file, width=10, height=8, useDingbats=FALSE)
for (outcome in outcomes_cols) {
	plt <- ggplot(meta_outcomes, aes_string(x=ga_col, fill=outcome)) +
		geom_bar(position="stack") +
		ggtitle(paste("Mednax", outcome, "distribution"))
	print(plt)

	# TODO: Can we do a positive vs. negative PPV for the outcome of interest
	# Where the classification is based on the majority class
	pv_df <- meta_outcomes %>%
		group_by(EGAbest) %>%
		mutate(
			allpos=1 %>% factor(levels=c(0, 1)),
			allneg=0 %>% factor(levels=c(0, 1)),
			majority=majority_label(get(outcome)) %>% factor(levels=c(0, 1)))

	onelabel_pvdf <- pv_df %>%
		summarize(
			npv_of_allzeros=npv_vec(get(outcome), allneg, event_level="second"),
			ppv_of_allones=ppv_vec(get(outcome), allpos, event_level="second")) %>%
		pivot_longer(-EGAbest, names_to="type", values_to="predictive_value")
	plt <- ggplot(onelabel_pvdf, aes(x=EGAbest, y=predictive_value, color=type)) +
		geom_line(size=2) +
		geom_point(size=5) +
		ylim(0, 1) +
		scale_color_manual(values=c(npv_of_allzeros="#457b9d", ppv_of_allones="#e63946")) +
		ggtitle(paste("Predictive Value One Label for", outcome))
	print(plt)

	majority_pvdf <- pv_df %>%
		summarize(
			npv=npv_vec(get(outcome), majority, event_level="second"),
			ppv=ppv_vec(get(outcome), majority, event_level="second")) %>%
		pivot_longer(-EGAbest, names_to="type", values_to="predictive_value")
	plt <- ggplot(majority_pvdf, aes(x=EGAbest, y=predictive_value, color=type)) +
		geom_line(size=2) +
		geom_point(size=5) +
		ylim(0, 1) +
		scale_color_manual(values=c(npv="#457b9d", ppv="#e63946")) +
		ggtitle(paste("Predictive Value Majority Label for", outcome))
	print(plt)
}
dev.off()

