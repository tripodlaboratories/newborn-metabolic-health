# Plot ROC curves using metabolites
library(dplyr)
library(readr)
library(data.table)
library(tidyr)
library(PRROC)
library(pROC)
library(ggthemes)
library(ggrepel)
library(ggplot2)

theme_set(theme_base(base_size=20))

# Helper Functions
get_auroc <- function(prediction_probs, true_labels, any_condition_col=NULL) {
	if (!is.null(any_condition_col)) {
		scores <- tibble(preds=prediction_probs, class_label=true_labels, any_condition=any_condition_col)
		scores <- scores %>% filter(!(class_label == 0 & any_condition != 0))
	} else {
		scores <- tibble(preds=prediction_probs, class_label=true_labels)
	}
  class_0_scores <- scores[scores$class_label == 0, ]$preds
  class_1_scores <- scores[scores$class_label == 1, ]$preds
  tryCatch({
		roc_results <- roc(
			controls=class_0_scores, cases=class_1_scores,
			direction="<")
  return(roc_results$auc %>% as.numeric())
	},
	error=function(cond) {
		return(NA)
	})
}

get_aupr <- function(prediction_probs, true_labels, any_condition_col=NULL) {
  # args:
  #   prediction_probs
  #   true_labels
  #   any_condition_col: encoded 1 if any condition is present, 0 otherwise
  # Filter cases that are negative for the current condition but have other conditions
	if (!is.null(any_condition_col)) {
		scores <- tibble(preds=prediction_probs, class_label=true_labels, any_condition=any_condition_col)
		scores <- scores %>% filter(!(class_label == 0 & any_condition != 0))
	} else {
		scores <- tibble(preds=prediction_probs, class_label=true_labels)
	}
	neg_label_scores <- scores[scores$class_label == 0, ]$preds
	pos_label_scores <- scores[scores$class_label == 1, ]$preds
  aupr <- pr.curve(pos_label_scores, neg_label_scores)
  return(aupr$auc.integral)
}

# Read in metabolite data
neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt") %>% sort()
outcomes_colors <- tableau_color_pal("Tableau 10")(length(neonatal_outcomes))
names(outcomes_colors) <- neonatal_outcomes

# Limit neonatal conditions to earlier gestational age range
gestational_age_ranges <- read_lines("./config/gestational_age_ranges.txt")
metadata <- fread("./data/processed/metadata.csv")
metadata <- metadata[gacat %in% gestational_age_ranges, ]

neonatal <- fread("./data/processed/neonatal_conditions.csv")
neonatal <- neonatal[row_id %in% metadata$row_id, ]
pos_fractions <- colSums(neonatal[, ..neonatal_outcomes]) / dim(neonatal)[1]
pos_fractions <- data.frame(outcome=names(pos_fractions), proportion_positive=pos_fractions)
write_csv(pos_fractions, "./results/neonatal/metabolite_scores/outcome_pos_frac.csv")

true_val_cols <- c("row_id", neonatal_outcomes)
neonatal[, num_conditions:=rowSums(.SD), .SDcols=neonatal_outcomes]
neonatal[, any_condition:=ifelse(num_conditions == 0, 0, 1)]
neonatal_with_overlap <- neonatal %>% mutate(
	neonatal_condition=case_when(
		nec_any == 1 & num_conditions == 1 ~ "NEC",
		rop_any == 1 & num_conditions == 1 ~ "ROP",
		bpd_any == 1 & num_conditions == 1 ~ "BPD",
		ivh_any == 1 & num_conditions == 1 ~ "IVH",
		num_conditions > 1 ~ "Overlapping Conditions",
		TRUE ~ "None of the Above"))

# Group by neonatal outcomes
sparse_cols <- colnames(neonatal)[colSums(is.na(neonatal)) > 5000]
neonatal <- select(neonatal, -one_of(sparse_cols), -row_id) %>% drop_na()
metabolites <- colnames(neonatal %>% select(-one_of(neonatal_outcomes)))
metabolites <- metabolites[!metabolites %in% c("num_conditions", "any_condition")]
neonatal_tall <- neonatal %>%
	select(-num_conditions) %>%
	gather("neonatal_condition", "condition_value", -metabolites, -any_condition) %>%
	gather("metabolite", "metabolite_value", -any_condition, -neonatal_condition, -condition_value)
metabolite_scores <- neonatal_tall %>%
	group_by(neonatal_condition, metabolite) %>%
	summarize(
		auroc=get_auroc(metabolite_value, condition_value, any_condition),
		aupr=get_aupr(metabolite_value, condition_value, any_condition))
neonatal_auroc <- metabolite_scores %>%
	select(-aupr) %>%
	spread(neonatal_condition, auroc)
neonatal_aupr <- metabolite_scores %>%
	select(-auroc) %>%
	spread(neonatal_condition, aupr)


outcome_comparisons <- list(
	c("x"="bpd_any", "y"="ivh_any"),
	c("x"="bpd_any", "y"="nec_any"),
	c("x"="bpd_any", "y"="rop_any"),
	c("x"="ivh_any", "y"="nec_any"),
	c("x"="ivh_any", "y"="rop_any"),
	c("x"="nec_any", "y"="rop_any")
)

pdf("./results/neonatal/plots/metabolite_auroc.pdf", useDingbats=FALSE)
for (coords in outcome_comparisons) {
	plt <- ggplot(neonatal_auroc, aes_string(coords[["x"]], coords[["y"]], label="metabolite")) +
		geom_point(size=4) +
		geom_hline(yintercept=0) +
		geom_vline(xintercept=0) +
		geom_abline(slope=1, intercept=0, linetype="dashed") +
		geom_text_repel() +
		xlab(paste0(coords[["x"]], "_auroc")) +
		ylab(paste0(coords[["y"]], "_auroc")) +
		ggtitle("AUROC")
	print(plt)
	plt <- ggplot(neonatal_aupr, aes_string(coords[["x"]], coords[["y"]], label="metabolite")) +
		geom_point(size=4) +
		geom_hline(yintercept=0) +
		geom_vline(xintercept=0) +
		geom_abline(slope=1, intercept=0, linetype="dashed") +
		geom_text_repel() +
		xlab(paste0(coords[["x"]], "_aupr")) +
		ylab(paste0(coords[["y"]], "_aupr")) +
		ggtitle("AUPR")
	print(plt)
}
dev.off()

write_csv(neonatal_auroc, "./results/neonatal/metabolite_scores/neonatal_metabolite_auroc.csv")
write_csv(neonatal_aupr, "./results/neonatal/metabolite_scores/neonatal_metabolite_aupr.csv")

# Repeat for cases only
neonatal_tall <- neonatal %>%
	filter(num_conditions >= 1) %>%
	select(-num_conditions) %>%
	gather("neonatal_condition", "condition_value", -metabolites, -any_condition) %>%
	gather("metabolite", "metabolite_value", -any_condition, -neonatal_condition, -condition_value)
metabolite_scores <- neonatal_tall %>%
	group_by(neonatal_condition, metabolite) %>%
	summarize(
		auroc=get_auroc(metabolite_value, condition_value, any_condition=NULL),
		aupr=get_aupr(metabolite_value, condition_value, any_condition=NULL))
neonatal_auroc <- metabolite_scores %>%
	select(-aupr) %>%
	spread(neonatal_condition, auroc)
neonatal_aupr <- metabolite_scores %>%
	select(-auroc) %>%
	spread(neonatal_condition, aupr)
pdf("./results/neonatal/plots/metabolite_auroc_cases_only.pdf", useDingbats=FALSE)
for (coords in outcome_comparisons) {
	x_coord <- coords[["x"]]
	y_coord <- coords[["y"]]
	plt <- ggplot(neonatal_auroc, aes_string(x_coord, y_coord, label="metabolite")) +
		geom_point(size=4) +
		geom_hline(yintercept=0) +
		geom_vline(xintercept=0) +
		geom_abline(slope=1, intercept=0, linetype="dashed") +
		geom_text_repel() +
		xlim(min(neonatal_auroc[[x_coord]]) - 0.075, max(neonatal_auroc[[x_coord]]) + 0.075) +
		ylim(min(neonatal_auroc[[y_coord]]) - 0.075, max(neonatal_auroc[[y_coord]]) + 0.075) +
		xlab(paste0(coords[["x"]], "_auroc")) +
		ylab(paste0(coords[["y"]], "_auroc")) +
		ggtitle("AUROC")
	print(plt)

	plt <- ggplot(neonatal_aupr, aes_string(coords[["x"]], coords[["y"]], label="metabolite")) +
		geom_point(size=4) +
		geom_hline(yintercept=0) +
		geom_vline(xintercept=0) +
		geom_abline(slope=1, intercept=0, linetype="dashed") +
		geom_text_repel() +
		xlim(min(neonatal_aupr[[x_coord]]) - 0.075, max(neonatal_aupr[[x_coord]]) + 0.075) +
		ylim(min(neonatal_aupr[[y_coord]]) - 0.075, max(neonatal_aupr[[y_coord]]) + 0.075) +
		xlab(paste0(coords[["x"]], "_aupr")) +
		ylab(paste0(coords[["y"]], "_aupr")) +
		ggtitle("AUPR")
	print(plt)
}
dev.off()

# Repeat for no overlap
neonatal_no_overlap <- neonatal %>% filter(num_conditions <= 1)
pos_fractions <- colSums(neonatal_no_overlap %>% select(one_of(neonatal_outcomes))) / dim(neonatal_no_overlap)[1]
pos_fractions <- data.frame(outcome=names(pos_fractions), proportion_positive=pos_fractions)
write_csv(pos_fractions, "./results/neonatal/metabolite_scores/outcome_pos_frac_no_overlap.csv")

neonatal_tall <- neonatal_no_overlap %>%
	select(-num_conditions) %>%
	gather("neonatal_condition", "condition_value", -metabolites, -any_condition) %>%
	gather("metabolite", "metabolite_value", -any_condition, -neonatal_condition, -condition_value)

metabolite_scores <- neonatal_tall %>%
	group_by(neonatal_condition, metabolite) %>%
	summarize(
		auroc=get_auroc(metabolite_value, condition_value, any_condition=NULL),
		aupr=get_aupr(metabolite_value, condition_value, any_condition=NULL))
neonatal_auroc <- metabolite_scores %>%
	select(-aupr) %>%
	spread(neonatal_condition, auroc)
neonatal_aupr <- metabolite_scores %>%
	select(-auroc) %>%
	spread(neonatal_condition, aupr)
pdf("./results/neonatal/plots/metabolite_auroc_no_overlap.pdf", useDingbats=FALSE)
for (coords in outcome_comparisons) {
	x_coord <- coords[["x"]]
	y_coord <- coords[["y"]]
	plt <- ggplot(neonatal_auroc, aes_string(x_coord, y_coord, label="metabolite")) +
		geom_point(size=4) +
		geom_hline(yintercept=0) +
		geom_vline(xintercept=0) +
		geom_abline(slope=1, intercept=0, linetype="dashed") +
		geom_text_repel() +
		xlim(min(neonatal_auroc[[x_coord]]) - 0.075, max(neonatal_auroc[[x_coord]]) + 0.075) +
		ylim(min(neonatal_auroc[[y_coord]]) - 0.075, max(neonatal_auroc[[y_coord]]) + 0.075) +
		xlab(paste0(coords[["x"]], "_auroc")) +
		ylab(paste0(coords[["y"]], "_auroc")) +
		ggtitle("AUROC")
	print(plt)

	plt <- ggplot(neonatal_aupr, aes_string(coords[["x"]], coords[["y"]], label="metabolite")) +
		geom_point(size=4) +
		geom_hline(yintercept=0) +
		geom_vline(xintercept=0) +
		geom_abline(slope=1, intercept=0, linetype="dashed") +
		geom_text_repel() +
		xlim(min(neonatal_aupr[[x_coord]]) - 0.075, max(neonatal_aupr[[x_coord]]) + 0.075) +
		ylim(min(neonatal_aupr[[y_coord]]) - 0.075, max(neonatal_aupr[[y_coord]]) + 0.075) +
		xlab(paste0(coords[["x"]], "_aupr")) +
		ylab(paste0(coords[["y"]], "_aupr")) +
		ggtitle("AUPR")
	print(plt)
}
dev.off()

write_csv(neonatal_auroc, "./results/neonatal/metabolite_scores/neonatal_metabolite_auroc_no_overlap.csv")
write_csv(neonatal_aupr, "./results/neonatal/metabolite_scores/neonatal_metabolite_aupr_no_overlap.csv")
