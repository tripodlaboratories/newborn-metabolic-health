# Plot distribution of single unit bottleneck outputs vs. predictions
library(optparse)
library(readr)
library(dplyr)
library(tidyr)
library(broom)
library(yardstick)
library(data.table)
library(stringr)
library(grid)
library(gridExtra)
library(ggridges)
library(ggplot2)
library(ggthemes)

theme_set(theme_base(base_size=18))

# Command line args
option_list <- list(
  make_option(c("-i", "--input_dir"),
              type="character",
              help="Input directory with bottleneck outputs and predictions."),
  make_option(c("-o", "--output_file"),
              type="character",
              help="Output file to save results."),
	make_option(c("--preds_filename"), type="character",
							help="filename for predictions file."),
	make_option(c("--true_vals_file"), type="character", default=NULL,
							help="Specify a file to match true values from."),
	make_option(c("--interactive_debug"), type="logical", default=FALSE,
							help="adds default options for debugging interactively"
	)
)

opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input_dir
output_file <- opts$output_file
preds_filename <- opts$preds_filename
true_vals_file <- opts$true_vals_file
interactive_debug <- opts$interactive_debug

if (isTRUE(interactive_debug)) {
	input_dir <- "./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/"
	preds_filename <- "bottleneck.csv"
	output_file <- "./.scratch/null_file.pdf"
}

# Read in predictions and true values
neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt") %>% sort()
outcomes_colors <- tableau_color_pal("Tableau 10")(length(neonatal_outcomes))
names(outcomes_colors) <- neonatal_outcomes

preds <- fread(paste0(input_dir, preds_filename))
setnames(preds, "V1", "row_id", skip_absent=TRUE)

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
preds[, (neonatal_outcomes):=bottleneck_unit_0]
colnames(preds) <- ifelse(
	colnames(preds) %in% neonatal_outcomes, paste0(colnames(preds), "_pred"),
	colnames(preds))
mean_preds <- preds[, lapply(.SD, mean), by=list(row_id),
										.SDcols=paste0(neonatal_outcomes, "_pred")]
true_vals[, total_conditions:=rowSums(.SD), .SDcols=neonatal_outcomes]
merged_results <- merge(mean_preds, true_vals, by="row_id")

# Plot stacked histogram of bottleneck unit outputs
pdf(output_file, height=7, width=7, useDingbats=FALSE)
for (outcome in neonatal_outcomes) {
	preds_col <- paste0(outcome, "_pred")
	hist_colors <- c("grey", outcomes_colors[[outcome]])
	names(hist_colors) <- c("0", outcome)
	plt <- ggplot(
		merged_results %>% mutate(recorded_outcome=ifelse(!!sym(outcome) == 0, "0", outcome)),
		aes_string(x=preds_col, fill="recorded_outcome")) +
		geom_histogram(bins=20, color="black") +
		scale_fill_manual(values=hist_colors) +
		xlab("bottleneck unit output") +
		ggtitle(paste0(outcome, " bottleneck as predictions."))
	print(plt)
}

metadata <- fread("./data/processed/metadata.csv")
neonatal_condition_colors <- c(
  "BPD"="#4E79A7", "IVH"="#F28E2B", "NEC"="#E15759", "ROP"="#76B7B2",
  "Multiple"="#027B8E", "Any"="#712b9c", "None"="light grey")
mean_preds <- preds[, lapply(.SD, mean), by=list(row_id),
										.SDcols=c("bottleneck_unit_0")]
merged_results <- merge(mean_preds, true_vals, by="row_id")
merged_results <- merge(merged_results, metadata, by="row_id") %>%
	mutate(
    neonatal_outcome=case_when(
      bpd_any == 1 & total_conditions == 1 ~ "BPD",
      ivh_any == 1 & total_conditions == 1 ~ "IVH",
      rop_any == 1 & total_conditions == 1 ~ "ROP",
      nec_any == 1 & total_conditions == 1 ~ "NEC",
			total_conditions > 1 ~ "Multiple",
      TRUE ~ "None") %>% factor(levels=names(neonatal_condition_colors)),
		any_outcome=case_when(
			total_conditions >= 1 ~ "Any",
			TRUE ~ "None") %>% factor(levels=names(neonatal_condition_colors))) %>%
	mutate(
		gest_age=gacat %>% gsub("^[0-9]{2}_", "", .) %>% as.numeric() - 0.5,
		birthweight=bwtcat %>% gsub("^[0-9]*_", "", .) %>% as.numeric() - 24
	)

# Make plots for gestational age and birth weight
plt <- ggplot(merged_results %>% mutate(gacat=gsub("_", "-", gacat)),
							aes(gacat, bottleneck_unit_0, fill=neonatal_outcome)) +
	geom_boxplot() +
	scale_fill_manual(name="Neonatal\nOutcome", values=neonatal_condition_colors) +
	xlab("Gestational Age (weeks)") +
	ylab("Bottleneck Unit Output")
print(plt)

# Repeat plot for `any_outcome`
plt <- ggplot(merged_results %>% mutate(gacat=gsub("_", "-", gacat)),
							aes(gacat, bottleneck_unit_0, fill=any_outcome)) +
	geom_boxplot() +
	xlab("Gestational Age (weeks)") +
	ylab("Bottleneck Unit Output") +
	scale_fill_manual(name="Neonatal\nOutcome", values=neonatal_condition_colors) +
	theme(axis.text=element_text(size=20), axis.title=element_text(size=24))
print(plt)
merged_results_and_outcomes <- merged_results

# Create correlation plot with bottleneck unit and additional covariates
source("./biobank_project/plotting/variable_correlations.R")
additional_covariates <- c(
	"gest_age", "birthweight", "bpd_any", "ivh_any",
  "nec_any", "rop_any", "mage_catm2", "bmi_catm", "cs_any", "prom_any",
	"ipi_cat", "Tocolysis_VS", "gdm", "dmtype1", "dmtype2", "ghtn", "spree",
	"superimposed", "parity_catm", "induce_any", "sga_any", "mortality")
spearman_vars <- c("bottleneck_unit_0", additional_covariates)
df_for_spearman <- merge(
		merged_results, fread("./data/processed/maternal_conditions.csv"), by="row_id") %>%
	mutate(
		sga_any=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0),
		mortality=ifelse(`_dthind` == 1, 1, 0)) %>%
	select(one_of(spearman_vars)) %>%
	filter(
		(mage_catm2 != 99) &
		(bmi_catm != 99) &
		(ipi_cat != 99 & !is.na(ipi_cat)))
plt <- create_spearman_plot(df_for_spearman, plot_title="co-morbidity spearman")
print(plt)

# Individual AUPR and AUROC curves for bottleneck layer vs. any across
# gestational ages
df_for_curves <- merged_results %>%
	select(gacat, bottleneck_unit_0, any_outcome) %>%
	mutate(any_outcome=factor(any_outcome),
				 gacat=gsub("_", "-", gacat))
if (which(levels(df_for_curves$any_outcome) == "None") == 1) {
	event_level_to_use <- "first"
} else {
	event_level_to_use <- "second"
}

curves_for_all_ga <- list()
for (ga in df_for_curves$gacat %>% sort() %>% unique()) {
	ga_df <- filter(df_for_curves, gacat == ga)

	roc_df <- roc_curve(
		ga_df, any_outcome, bottleneck_unit_0,
		event_level=event_level_to_use) %>%
	mutate(TPR=sensitivity, FPR=1 - specificity)
	roc_auc_value <- roc_auc(
		ga_df, any_outcome, bottleneck_unit_0,
		event_level=event_level_to_use)
	auroc_label <- paste0("AUC: ", round(roc_auc_value$.estimate, 3))
	plt <- ggplot(roc_df, aes(x=FPR, y=TPR)) +
		geom_path(color=neonatal_condition_colors[["Any"]]) +
		geom_segment(aes(x=0, xend=1, y=0, yend=1), color="grey", linetype="dashed") +
		ylim(0, 1) +
		annotate("text", x=0.75, y=0.2, size=10, label=auroc_label) +
		ggtitle(paste(ga, "ROC Curve")) +
		theme(aspect.ratio=1)
	print(plt)

	pr_df <- pr_curve(
		ga_df, any_outcome, bottleneck_unit_0,
		event_level=event_level_to_use) %>%
	rename(Recall=recall, Precision=precision)
	pr_df[["GA"]] <- ga
	pr_auc_value <- pr_auc(
		ga_df, any_outcome, bottleneck_unit_0,
		event_level=event_level_to_use)
	aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
	plt <- ggplot(pr_df, aes(x=Recall, y=Precision)) +
		geom_path(color=neonatal_condition_colors[["Any"]]) +
		ylim(0, 1) +
		annotate("text", x=0.8, y=1.0, size=10, label=aupr_label) +
		ggtitle(paste(ga, "PR-Curve")) +
		theme(aspect.ratio=1)
	print(plt)
	curves_for_all_ga <- append(curves_for_all_ga, list(pr_df))
}

# Create a combined plot for AUPR curves for `any_outcome`
large_plot_colors <- c(
	"22-23"="#b273d9",
	"24-25"="#9037c8",
	"26-27"="#5f2484",
	"28-29"="#37154c"
)
large_combined_plt <- ggplot(
	bind_rows(curves_for_all_ga),
	aes(x=Recall, y=Precision, color=GA)) +
	geom_path() +
	scale_color_manual(values=large_plot_colors) +
	ylim(0, 1) +
	ggtitle("Precision-Recall Curve") +
	labs(color="GA") +
	theme(aspect.ratio=1)
print(large_combined_plt)

# TODO: Plot a precision-recall curve for infants across all gestational age
# ranges
pr_df <- pr_curve(
	df_for_curves, any_outcome, bottleneck_unit_0,
	event_level=event_level_to_use) %>%
rename(Recall=recall, Precision=precision)
pr_auc_value <- pr_auc(
	df_for_curves, any_outcome, bottleneck_unit_0,
	event_level=event_level_to_use)
aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
plt <- ggplot(pr_df, aes(x=Recall, y=Precision)) +
	geom_path(color=neonatal_condition_colors[["Any"]]) +
	ylim(0, 1) +
	annotate("text", x=0.8, y=1.0, size=10, label=aupr_label) +
	ggtitle("PR-Curve (all GA)") +
	theme(aspect.ratio=1)
print(plt)

# TODO: Calculate a single unit bottleneck adjustment factor
# Based on median gestational age
bottleneck_medians <- df_for_curves %>%
	select(gacat, bottleneck_unit_0) %>%
	group_by(gacat) %>%
	dplyr::summarize(median_bottleneck=median(bottleneck_unit_0)) %>%
	mutate(gest_age=gsub("^[0-9]{2}-", "", gacat) %>% as.numeric() - 0.5)
# TODO: Try an exponential model
linear_fit <- lm(log(median_bottleneck) ~ gest_age, data=bottleneck_medians)
coefs <- linear_fit$coefficients
offset <- coefs[["(Intercept)"]]
adjustment_factor <- coefs[["gest_age"]]

df_for_curves <- df_for_curves %>%
	mutate(gest_age=gsub("^[0-9]{2}-", "", gacat) %>% as.numeric() - 0.5) %>%
	mutate(adj_bottleneck=bottleneck_unit_0 + exp(gest_age * adjustment_factor + offset))

# Plot the impact of bottleneck adjustment
plt <- ggplot(df_for_curves, aes(bottleneck_unit_0, adj_bottleneck, color=gest_age)) +
	geom_point()
print(plt)

# Plot AUPR curves for comparison
pr_df <- pr_curve(
	df_for_curves, any_outcome, adj_bottleneck,
	event_level=event_level_to_use) %>%
rename(Recall=recall, Precision=precision)
pr_auc_value <- pr_auc(
	df_for_curves, any_outcome, adj_bottleneck,
	event_level=event_level_to_use)
aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
plt <- ggplot(pr_df, aes(x=Recall, y=Precision)) +
	geom_path(color=neonatal_condition_colors[["Any"]]) +
	ylim(0, 1) +
	annotate("text", x=0.8, y=1.0, size=10, label=aupr_label) +
	ggtitle("PR-Curve (all GA), adjusted bottleneck") +
	theme(aspect.ratio=1)
print(plt)

# Create AUPR/AUROC curves on a per outcome level
df_for_curves <- merged_results %>%
	select(gacat, bottleneck_unit_0, one_of(neonatal_outcomes), total_conditions) %>%
	mutate(gacat=gsub("_", "-", gacat))
event_level_to_use <- "first"

ga_ranges <- df_for_curves$gacat %>% sort() %>% unique()
curves_for_all_ga <- vector("list")

for (ga in ga_ranges) {
	ga_df <- filter(df_for_curves, gacat == ga)

	prcurve_list <- vector("list", length(neonatal_outcomes))
	names(prcurve_list) <- neonatal_outcomes
	aupr_list <- vector("character", length(neonatal_outcomes))
	names(aupr_list) <- neonatal_outcomes
	for (outcome in neonatal_outcomes) {
		df_for_roc <- ga_df %>%
			filter(!(get(outcome) == 0 & total_conditions >= 1))
		pr_df <- pr_curve(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, bottleneck_unit_0, event_level=event_level_to_use) %>%
			rename(Recall=recall, Precision=precision)
		pr_auc_value <- pr_auc(
			df_for_roc %>% mutate(!!as.name(outcome) := factor(!!as.name(outcome))),
			!!outcome, bottleneck_unit_0, event_level=event_level_to_use)

		outcome_label <- outcome %>% gsub("_any", "", .) %>% toupper()
		pr_df[["Neonatal\nOutcome"]] <- outcome_label
		pr_df[["Neonatal\nOutcome\n(GA)"]] <- paste0(outcome_label, " (", ga, ")")
		prcurve_list[[outcome]] <- pr_df
		aupr_label <- paste0("AUC: ", round(pr_auc_value$.estimate, 3))
		aupr_list[[outcome]] <- aupr_label
	}
	combined_plt <- ggplot(bind_rows(prcurve_list), aes(x=Recall, y=Precision, color=`Neonatal\nOutcome`)) +
		geom_path() +
		scale_color_tableau() +
		ylim(0, 1) +
		annotate("text", x=0.5, y=0.3, size=6, color=neonatal_condition_colors[["BPD"]],
						label=paste("BPD", aupr_list[["bpd_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.25, size=6, color=neonatal_condition_colors[["IVH"]],
						label=paste("IVH", aupr_list[["ivh_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.2, size=6, color=neonatal_condition_colors[["NEC"]],
						label=paste("NEC", aupr_list[["nec_any"]]), hjust=0) +
		annotate("text", x=0.5, y=0.15, size=6, color=neonatal_condition_colors[["ROP"]],
						label=paste("ROP", aupr_list[["rop_any"]]), hjust=0) +
		ggtitle(paste(ga, "Precision-Recall Curve")) +
		labs(color="Neonatal\nOutcome") +
		theme(aspect.ratio=1)
	print(combined_plt)
	curves_for_all_ga <- append(curves_for_all_ga, prcurve_list)
}

# Create combined gestational age AUPR curves
large_plot_colors <- c(
	"BPD (22-23)"="#5581af",
	"BPD (24-25)"="#43678e",
	"BPD (26-27)"="#34516f",
	"BPD (28-29)"="#293f57",
	"IVH (22-23)"="#f39c44",
	"IVH (24-25)"="#f08214",
	"IVH (26-27)"="#ce6e0d",
	"IVH (28-29)"="#99520a",
	"NEC (22-23)"="#e05254",
	"NEC (24-25)"="#d92629",
	"NEC (26-27)"="#981b1d",
	"NEC (28-29)"="#6c1315",
	"ROP (22-23)"="#69b0aa",
	"ROP (24-25)"="#45827d",
	"ROP (26-27)"="#2c5450",
	"ROP (28-29)"="#203c3a"
)
large_combined_plt <- ggplot(
		bind_rows(curves_for_all_ga),
		aes(x=Recall, y=Precision, color=`Neonatal\nOutcome\n(GA)`)) +
		geom_path() +
		scale_color_manual(values=large_plot_colors) +
		ylim(0, 1) +
		ggtitle("Precision-Recall Curve") +
		labs(color="Neonatal\nOutcome\n(GA)") +
		theme(aspect.ratio=1)
	print(large_combined_plt)

# Nested ANOVA with individual comparisons
df_for_anova <- merged_results %>%
	select(bottleneck_unit_0, neonatal_outcome, gacat)
anova_summary_df <- df_for_anova %>% group_by(gacat) %>%
	do(tidy(aov(bottleneck_unit_0 ~ neonatal_outcome, data=.)))
grid.newpage()
grid.table(anova_summary_df, theme=ttheme_default(base_size=11))

run_posthoc_t_test <- function(df, outcome_colname, p_adjust_method="BH") {
	full_pairwise <- pairwise.t.test(df$bottleneck_unit_0, df[[outcome_colname]], p.adjust.method="none")
	comparisons_of_interest <- full_pairwise$p.value["None", ]
	p_vals <- tibble(none_vs=names(comparisons_of_interest), p_val=comparisons_of_interest) %>%
		mutate(adj_p_val=p.adjust(p_val, p_adjust_method))
}
t_test_results <- lapply(
	split(df_for_anova, df_for_anova$gacat), run_posthoc_t_test, outcome_colname="neonatal_outcome")
posthoc_t_test_df <- rbindlist(t_test_results, idcol=TRUE) %>%
	rename(gacat=`.id`)
grid.newpage()
grid.table(posthoc_t_test_df)

# Repeat ANOVA for any outcome
df_for_anova <- merged_results %>%
	select(bottleneck_unit_0, any_outcome, gacat)
anova_summary_df <- df_for_anova %>% group_by(gacat) %>%
	do(tidy(aov(bottleneck_unit_0 ~ any_outcome, data=.)))
grid.newpage()
grid.table(anova_summary_df, theme=ttheme_default(base_size=11))
run_posthoc_t_test <- function(df, outcome_colname, p_adjust_method="BH") {
	full_pairwise <- pairwise.t.test(df$bottleneck_unit_0, df[[outcome_colname]], p.adjust.method="none")
	comparisons_of_interest <- full_pairwise$p.value
	p_vals <- tibble(none_vs=colnames(comparisons_of_interest), p_val=comparisons_of_interest) %>%
		mutate(adj_p_val=p.adjust(p_val, p_adjust_method))
}
t_test_results <- lapply(
	split(df_for_anova, df_for_anova$gacat), run_posthoc_t_test, outcome_colname="any_outcome")
posthoc_t_test_df <- rbindlist(t_test_results, idcol=TRUE) %>%
	rename(gacat=`.id`)
grid.newpage()
grid.table(posthoc_t_test_df)

# Create a barplot of p values
posthoc_t_test_df <- posthoc_t_test_df %>%
	mutate(`-Log10 P Value`=-log10(p_val), gacat=gsub("_", "-", gacat))
plt <-ggplot(posthoc_t_test_df, aes(gacat, `-Log10 P Value`)) +
	geom_col(fill="white", color="black", width=0.7) +
	geom_hline(yintercept=-log10(0.05), color="#9c2b2b", linetype="dashed", size=1.2) + # p value
	geom_hline(yintercept=-log10(0.05 / 4), color="#2b439c", linetype="dashed", size=1.2) + # Bonferroni-corrected p value
	xlab("Gestational Age (weeks)") +
	theme(axis.text.y=element_text(size=18),
				axis.text.x=element_text(size=18, angle=90, vjust=0.5, hjust=1),
				axis.title=element_text(size=18),
				aspect.ratio=2)
print(plt)

# Plot bottleneck unit output faceted by neonatal outcome
plt <- ggplot(merged_results, aes(gest_age, bottleneck_unit_0, color=neonatal_outcome)) +
		geom_point(position=position_jitter(seed=100, width=0.4), size=2.5, alpha=0.6) +
		scale_color_manual(values=neonatal_condition_colors) +
		facet_wrap(~ neonatal_outcome, ncol=2)
print(plt)
plt <- ggplot(merged_results, aes(birthweight, bottleneck_unit_0, color=neonatal_outcome)) +
	geom_point(size=2.5, alpha=0.6) +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~ neonatal_outcome, ncol=2)
print(plt)

# Plot various metrics related to performance at each gestational age.
get_performance <- function(
	pred_value, outcome_value, total_conditions, metric_fn, rev_preds_direction) {
		if (isTRUE(rev_preds_direction)) {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}
		results_df <- data.frame(
			pred=pred_value, outcome=outcome_value, total_conditions=total_conditions) %>%
		filter(!(outcome == 0 & total_conditions > 0)) %>%
		mutate(outcome=factor(outcome)) %>%
		metric_fn(outcome, pred, event_level=event_level_to_use)
		return(results_df$.estimate)
	}

rev_preds_direction <- TRUE
preds_true_vals <- merge(
		preds %>% select(row_id, iter, contains("_pred")) %>%
			gather("outcome", "pred_value", -row_id, -iter) %>%
			mutate(outcome=gsub("_any_pred", "", outcome) %>% toupper()),
		true_vals %>% select(row_id, total_conditions, contains("_any")) %>%
			gather("outcome", "outcome_value", -row_id, -total_conditions) %>%
			mutate(outcome=gsub("_any", "", outcome) %>% toupper()),
		by=c("row_id", "outcome"))
preds_true_vals <- merge(preds_true_vals, metadata[, list(row_id, gacat, bwtcat)], by="row_id") %>%
	mutate(gacat=gsub("_", "-", gacat))
classifier_scores <-  preds_true_vals %>%
		group_by(gacat, outcome, iter) %>%
		mutate(
			aupr=get_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=pr_auc, rev_preds_direction=rev_preds_direction),
			auroc=get_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=roc_auc, rev_preds_direction=rev_preds_direction),
			average_precision=get_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=average_precision, rev_preds_direction=rev_preds_direction)) %>%
		select(outcome, iter, aupr, auroc, average_precision) %>%
		distinct()

theme_set(theme_base(base_size=16))
plt <- ggplot(classifier_scores, aes(gacat, aupr, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age")
print(plt)

plt <- ggplot(classifier_scores, aes(gacat, auroc, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age")
print(plt)

plt <- ggplot(classifier_scores, aes(gacat, average_precision, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age")
print(plt)

# TODO: Modify function to return thresholds themselves for plotting?
# Use various percentile thresholds to calculate metrics
get_threshold_performance <- function(
			pred_value, outcome_value, total_conditions, metric_fn, rev_preds_direction) {
		if (isTRUE(rev_preds_direction)) {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}
		results_df <- data.frame(
			pred=pred_value, outcome=outcome_value, total_conditions=total_conditions) %>%
		filter(!(outcome == 0 & total_conditions > 0)) %>%
		mutate(outcome=factor(outcome))

		percentile_thresholds <- c(.1, .2, .3, .4, .5, .6, .7, .8, .9)
		scores_over_thresholds <- c()
		threshold_values <- c()
		for (threshold in percentile_thresholds) {
			threshold_value <- quantile(results_df[["pred"]], threshold)
			if (isTRUE(rev_preds_direction)) {
				threshold_df <- results_df %>%
					mutate(prediction_label=ifelse(pred < threshold_value, 1, 0) %>% factor())
			} else {
				threshold_df <- results_df %>%
					mutate(prediction_label=ifelse(pred >= threshold_value, 1, 0) %>% factor())
			}
			if (length(levels(threshold_df$prediction_label)) < length(levels(threshold_df$outcome))) {
				next
			}
			current_score <- metric_fn(threshold_df, outcome, prediction_label)
			scores_over_thresholds <- c(scores_over_thresholds, current_score$.estimate)
			threshold_values <- c(threshold_values, threshold_value)
		}
		return(list(
			"scores"=max(scores_over_thresholds),
			"thresholds"=threshold_values[which(scores_over_thresholds == max(scores_over_thresholds))]
			))
	}

classifier_scores <-  preds_true_vals %>%
		group_by(gacat, outcome, iter) %>%
		mutate(
			max_npv=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=npv, rev_preds_direction=rev_preds_direction)$scores,
			max_specificity=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=specificity, rev_preds_direction=rev_preds_direction)$scores,
			max_sensitivity=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=sensitivity, rev_preds_direction=rev_preds_direction)$scores) %>%
		select(outcome, iter, max_npv, max_specificity, max_sensitivity) %>%
		distinct()
plt <- ggplot(classifier_scores, aes(gacat, max_npv, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age")
print(plt)

plt <- ggplot(classifier_scores, aes(gacat, max_sensitivity, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age")
print(plt)

plt <- ggplot(classifier_scores, aes(gacat, max_specificity, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age")
print(plt)

# TODO: Plot thresholds for any outcome adjusted by gestational age
# Calculate thresholds for any outcome vs. none
preds_true_vals_any <- merge(
		preds %>% mutate(any_outcome_pred=bottleneck_unit_0) %>%
			select(row_id, iter, any_outcome_pred) %>%
			gather("outcome", "pred_value", -row_id, -iter) %>%
			mutate(outcome=gsub("_pred", "", outcome)),
		true_vals %>% mutate(any_outcome=ifelse(total_conditions > 0, 1, 0)) %>%
			select(row_id, any_outcome, total_conditions) %>%
			gather("outcome", "outcome_value", -row_id, -total_conditions),
		by=c("row_id", "outcome"))
preds_true_vals_any <- merge(
	preds_true_vals_any, metadata[, list(row_id, gacat, bwtcat)], by="row_id") %>%
	mutate(gacat=gsub("_", "-", gacat))
any_outcome_scores <-  preds_true_vals_any %>%
		group_by(gacat, outcome, iter) %>%
		mutate(
			max_precision=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=precision, rev_preds_direction=rev_preds_direction)$scores,
			max_recall=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=recall, rev_preds_direction=rev_preds_direction)$scores,
			max_npv=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=npv, rev_preds_direction=rev_preds_direction)$scores,
			max_specificity=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=specificity, rev_preds_direction=rev_preds_direction)$scores,
			max_sensitivity=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=sensitivity, rev_preds_direction=rev_preds_direction)$scores) %>%
		select(outcome, iter, max_precision, max_recall, max_npv, max_specificity, max_sensitivity) %>%
		distinct()
any_outcome_thresholds <- preds_true_vals_any %>%
		group_by(gacat, outcome, iter) %>%
		mutate(
			max_precision=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=precision, rev_preds_direction=rev_preds_direction)$thresholds,
			max_recall=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=recall, rev_preds_direction=rev_preds_direction)$thresholds,
			max_npv=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=npv, rev_preds_direction=rev_preds_direction)$thresholds,
			max_specificity=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=specificity, rev_preds_direction=rev_preds_direction)$thresholds,
			max_sensitivity=get_threshold_performance(
				pred_value, outcome_value, total_conditions,
				metric_fn=sensitivity, rev_preds_direction=rev_preds_direction)$thresholds) %>%
		select(outcome, iter, max_precision, max_recall, max_npv, max_specificity, max_sensitivity) %>%
		distinct()
mean_thresholds <- any_outcome_thresholds %>% ungroup() %>% select(-iter) %>%
	group_by(gacat, outcome) %>% summarize_all(mean)
thresholds_for_plotting <- mean_thresholds %>% ungroup() %>%
	mutate(
		gacat_numeric=gacat %>% as.factor() %>% as.numeric(),
		x=gacat_numeric-0.4, xend=gacat_numeric+0.4)

# TODO: Plot thresholds over previous plot
plt <- ggplot(merged_results_and_outcomes %>% mutate(gacat=gsub("_", "-", gacat)),
							aes(gacat, bottleneck_unit_0, color=any_outcome)) +
	geom_point(position=position_jitterdodge(jitter.width=1.2, dodge.width=0.75, seed=100), alpha=0.6) +
	geom_segment(
		data=thresholds_for_plotting,
		aes(x=x, xend=xend, y=max_precision, yend=max_precision),
		color="black", linetype="dashed", size=0.8, inherit.aes=FALSE) +
	xlab("Gestational Age (weeks)") +
	ylab("Bottleneck Unit Output") +
	scale_color_manual(name="Neonatal\nOutcome", values=neonatal_condition_colors) +
	theme(axis.text=element_text(size=20), axis.title=element_text(size=24))
print(plt)

# Test a max-fmeasure model and calculate sensitivity/specificity
test_max_fmeasure <- function(pred_value, outcome_value, total_conditions, metric_fn, rev_preds_direction) {
		if (isTRUE(rev_preds_direction)) {
			event_level_to_use <- "first"
		} else {
			event_level_to_use <- "second"
		}
		results_df <- data.frame(
			pred=pred_value, outcome=outcome_value, total_conditions=total_conditions) %>%
		filter(!(outcome == 0 & total_conditions > 0)) %>%
		mutate(outcome=factor(outcome))

		percentile_thresholds <- c(.1, .2, .3, .4, .5, .6, .7, .8, .9)
		scores_over_thresholds <- c()
		threshold_vals <- c()
		threshold_dfs <- list()
		for (threshold in percentile_thresholds) {
			threshold_value <- quantile(results_df[["pred"]], threshold)
			if (isTRUE(rev_preds_direction)) {
				threshold_df <- results_df %>%
					mutate(prediction_label=ifelse(pred < threshold_value, 1, 0) %>% factor())
			} else {
				threshold_df <- results_df %>%
					mutate(prediction_label=ifelse(pred >= threshold_value, 1, 0) %>% factor())
			}
			if (length(levels(threshold_df$prediction_label)) < length(levels(threshold_df$outcome))) {
				next
			}
			current_score <- f_meas(threshold_df, outcome, prediction_label)
			scores_over_thresholds <- c(scores_over_thresholds, current_score$.estimate)
			threshold_vals <- c(threshold_vals, threshold_value)
			threshold_dfs <- append(threshold_dfs, threshold_df)
		}

		max_score <- max(scores_over_thresholds)
		corresponding_df <- threshold_dfs[which(scores_over_thresholds == max_score)]
		eval_score <- metric_fn(threshold_df, outcome, prediction_label)$.estimate
	}
fmeasure_scores <-  preds_true_vals %>%
		group_by(gacat, outcome, iter) %>%
		mutate(
			max_specificity=test_max_fmeasure(
				pred_value, outcome_value, total_conditions,
				metric_fn=specificity, rev_preds_direction=rev_preds_direction),
			max_sensitivity=test_max_fmeasure(
				pred_value, outcome_value, total_conditions,
				metric_fn=sensitivity, rev_preds_direction=rev_preds_direction)) %>%
		select(outcome, iter, max_specificity, max_sensitivity) %>%
		distinct()

plt <- ggplot(fmeasure_scores, aes(gacat, max_sensitivity, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age") +
	ggtitle("F-Measure Thresholds")
print(plt)

plt <- ggplot(fmeasure_scores, aes(gacat, max_specificity, color=outcome)) +
	geom_boxplot() +
	scale_color_manual(values=neonatal_condition_colors) +
	facet_wrap(~outcome, ncol=2) +
	xlab("Gestational Age") +
	ggtitle("F-Measure Thresholds")
print(plt)

# Evaluate using gestational age and birth weight as well
# Probably just a bar chart with single values (not grouped?)
# Data structure will have to be slightly different
gabwt_outcomes <- merge(
		merged_results %>% select(row_id, gest_age, birthweight),
		true_vals %>% select(row_id, total_conditions, contains("_any")) %>%
			gather("outcome", "outcome_value", -row_id, -total_conditions) %>%
			mutate(outcome=gsub("_any", "", outcome) %>% toupper()),
		by=c("row_id"))
fmeasure_gabwt <- gabwt_outcomes %>%
	group_by(outcome) %>%
	mutate(
			max_specificity_gest_age=test_max_fmeasure(
				gest_age, outcome_value, total_conditions,
				metric_fn=specificity, rev_preds_direction=rev_preds_direction),
			max_sensitivity_gest_age=test_max_fmeasure(
				gest_age, outcome_value, total_conditions,
				metric_fn=sensitivity, rev_preds_direction=rev_preds_direction),
			max_specificity_birthweight=test_max_fmeasure(
				birthweight, outcome_value, total_conditions,
				metric_fn=specificity, rev_preds_direction=rev_preds_direction),
			max_sensitivity_birthweight=test_max_fmeasure(
				birthweight, outcome_value, total_conditions,
				metric_fn=sensitivity, rev_preds_direction=rev_preds_direction)) %>%
		select(outcome, max_specificity_gest_age, max_sensitivity_gest_age, max_specificity_birthweight, max_sensitivity_birthweight) %>%
		distinct()

plt <- ggplot(fmeasure_gabwt %>% gather("metric", "value", -outcome), aes(outcome, value, fill=outcome)) +
	geom_col() +
	scale_fill_manual(values=neonatal_condition_colors) +
	facet_wrap(~metric, ncol=2) +
	ggtitle("F-Measure Thresholds With GA + Birthweight") +
	theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1))
print(plt)

# Use gestational age and birthweight as continuous metrics
gabwt_scores <- gabwt_outcomes %>%
	group_by(outcome) %>%
	mutate(
		average_precision_gest_age=get_performance(
				gest_age, outcome_value, total_conditions,
				metric_fn=average_precision, rev_preds_direction=rev_preds_direction),
		average_precision_birthweight=get_performance(
				birthweight, outcome_value, total_conditions,
				metric_fn=average_precision, rev_preds_direction=rev_preds_direction)) %>%
		select(outcome, average_precision_gest_age, average_precision_birthweight) %>%
		distinct()
plt <- ggplot(gabwt_scores %>% gather("metric", "value", -outcome), aes(outcome, value, fill=outcome)) +
	geom_col() +
	scale_fill_manual(values=neonatal_condition_colors) +
	facet_wrap(~metric, ncol=2) +
	ggtitle("Scores Using GA + Birthweight") +
	theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1))
print(plt)
dev.off()

