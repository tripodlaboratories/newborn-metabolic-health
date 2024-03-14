# Plot changes in correlation for metabolites in neonatal outcomes vs. controls
library(dplyr)
library(readr)
library(data.table)
library(tidyr)
library(caret)
library(pROC)
library(grid)
library(gridExtra)
library(WGCNA)
library(yardstick)
library(assertthat)
library(GGally)
library(ggthemes)
library(ggplot2)
library(ggrepel)
library(ggcorrplot)
library(tidygraph)
library(ggraph)

theme_set(theme_base(base_size=24))

# Read in neonatal conditions and metabolites
input_file <- "./data/processed/neonatal_conditions.csv"
neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt") %>% sort()
neonatal_outcome_colors <- c(
  "BPD"="#4E79A7", "IVH"="#F28E2B", "NEC"="#E15759", "ROP"="#76B7B2",
  "Multiple"="#027B8E", "None"="light grey")
input_data <- fread(input_file) %>% na_if("NULL")

# Subset on gestational ages at the extreme preterm end
ga_ranges <- read_lines("./config/gestational_age_ranges.txt")
metadata <- fread("./data/processed/metadata.csv")
ga_range_ids <- metadata %>% filter(gacat %in% ga_ranges)
input_data <- input_data %>% filter(row_id %in% ga_range_ids[["row_id"]])

outcome_data <- input_data %>% select(one_of(c("row_id", neonatal_outcomes))) %>%
  mutate(num_conditions=rowSums(.[, ..neonatal_outcomes])) %>%
  mutate(
    neonatal_outcome=case_when(
      bpd_any == 1 & num_conditions == 1 ~ "BPD",
      ivh_any == 1 & num_conditions == 1 ~ "IVH",
      rop_any == 1 & num_conditions == 1 ~ "ROP",
      nec_any == 1 & num_conditions == 1 ~ "NEC",
      num_conditions > 1 ~ "Multiple",
      TRUE ~ "None") %>% factor(levels=names(neonatal_outcome_colors))
  )
features <- input_data %>% select(-one_of(neonatal_outcomes))

# Drop sparse columns
non_sparse_cols <- colnames(features)[colSums(!is.na(features)) > (nrow(features) / 2)]
features <- features[, ..non_sparse_cols] %>% drop_na()

# Read in metabolite labels and replace columns
metabolite_labels <- fread("./config/metabolite_labels.csv")
setnames(
	features, metabolite_labels$raw_feature_name,
	metabolite_labels$feature_label, skip_absent=TRUE)


# Store cases, controls, correlation matrix, and correlation dataframe for
# each neonatal outcome
results <- list(list(), list(), list(), list())
names(results) <- neonatal_outcomes

# Iterate over neonatal outcomes
pdf("./results/correlation_disruption.pdf", width=7, height=7, useDingbats=FALSE)
for (outcome in neonatal_outcomes) {
	# Get correlations between metabolites in infants with each outcome and
	# infants without each outcome
	cases <- outcome_data %>% filter(get(outcome) == 1)
	controls <- outcome_data %>% filter(get(outcome) == 0 & num_conditions == 0)
	assertion <- assert_that(cases[[outcome]] %>% unique() == 1)
	assertion <- assert_that(controls[[outcome]] %>% unique() == 0)
	results[[outcome]][["case_metabolites"]] <- features %>% filter(row_id %in% cases[["row_id"]])
	results[[outcome]][["control_metabolites"]] <- features %>% filter(row_id %in% controls[["row_id"]])

	case_correlations <- features %>% filter(row_id %in% cases[["row_id"]]) %>%
		select(-row_id) %>% data.matrix() %>% WGCNA::cor(method="spearman", nThreads=8)
	control_correlations <- features %>% filter(row_id %in% controls[["row_id"]]) %>%
		select(-row_id) %>% data.matrix() %>% WGCNA::cor(method="spearman", nThreads=8)

	# Plot correlation matrices
	# Calculate clusters on the control matrix
	# See how the same hierarchical clustering order is disrupted in cases
	control_clust <- hclust(dist(control_correlations))
	control_clustered <- control_correlations[control_clust$order, control_clust$order]
	case_by_control_clust <- case_correlations[control_clust$order, control_clust$order]
	# Set up correlation plotting parameters

	plt <- ggcorrplot(
		control_clustered, hc.order=FALSE, lab=FALSE,
		colors=c("#6D9EC1", "white", "#EF5350"),
		show.diag=TRUE, ggtheme=ggthemes::theme_base(base_size=12),
		tl.cex=6) +
		ggtitle(paste0(outcome, " control correlation"))
	print(plt)
	plt <- ggcorrplot(
		control_clustered %>% unname(), hc.order=FALSE, lab=FALSE,
		colors=c("#6D9EC1", "white", "#EF5350"), outline.color=NA,
		show.diag=TRUE, ggtheme=ggthemes::theme_base(base_size=12)) +
		ggtitle(paste0(outcome, " control correlation"))
	print(plt)
	plt <- ggcorrplot(
		case_by_control_clust, hc.order=FALSE, lab=FALSE,
		colors=c("#6D9EC1", "white", "#EF5350"),
		show.diag=TRUE, ggtheme=ggthemes::theme_base(base_size=12),
		tl.cex=6) +
		ggtitle(paste0(outcome, " case correlation"))
	print(plt)
	plt <- ggcorrplot(
		case_by_control_clust %>% unname(), hc.order=FALSE, lab=FALSE,
		colors=c("#6D9EC1", "white", "#EF5350"), outline.color=NA,
		show.diag=TRUE, ggtheme=ggthemes::theme_base(base_size=12)) +
		theme(axis.text=element_blank()) +
		ggtitle(paste0(outcome, " case correlation"))
	print(plt)

	# Organize correlation matrices in dataframe form to compare as scatterplots
	create_corr_df <- function(correlation_matrix) {
		upper_tri_idx <- which(upper.tri(correlation_matrix, diag=FALSE), arr.ind=TRUE)
		case_corr_df <- tibble(
			column=dimnames(correlation_matrix)[[2]][upper_tri_idx[, 2]],
			row=dimnames(correlation_matrix)[[1]][upper_tri_idx[, 1]],
			spearman=correlation_matrix[upper_tri_idx]) %>%
		unite(metabolite_pair, sep=",", column, row)
	}

	case_corr_df <- create_corr_df(case_correlations) %>%
		rename(case_spearman_rho=spearman) %>%
		mutate(case_pval=WGCNA::corPvalueFisher(case_spearman_rho, nSamples=nrow(cases))) %>%
		mutate(case_log10_pval=-log10(case_pval)) %>%
		mutate(case_log10_pval=ifelse(
			is.infinite(case_log10_pval),
			max(case_log10_pval[!is.infinite(case_log10_pval)]),
			case_log10_pval))
	control_corr_df <- create_corr_df(control_correlations) %>%
		rename(control_spearman_rho=spearman) %>%
		mutate(control_pval=WGCNA::corPvalueFisher(control_spearman_rho, nSamples=nrow(controls))) %>%
		mutate(control_log10_pval=-log10(control_pval)) %>%
		mutate(control_log10_pval=ifelse(
			is.infinite(control_log10_pval),
			max(control_log10_pval[!is.infinite(control_log10_pval)]),
			control_log10_pval))

	corr_df <- merge(case_corr_df, control_corr_df, by="metabolite_pair") %>%
		mutate(abs_pval_diff=abs(case_log10_pval - control_log10_pval))
	pval_diff_threshold <- quantile(corr_df$abs_pval_diff, .95, na.rm=TRUE)
	corr_df <- corr_df %>%
		mutate(
			metabolite_pair_label=ifelse(
				abs_pval_diff > pval_diff_threshold,
				metabolite_pair, "")
		)
	results[[outcome]][["corr_df"]] <- corr_df

	# Plot a scatterplot comparing the scenarios: each point is a pair of
	# metabolites in cases and controls.
	neonatal_outcome_title <- gsub("_any", "", outcome) %>% toupper()
	plt <- ggplot(corr_df, aes(control_log10_pval, case_log10_pval)) +
		geom_hline(yintercept=0) +
		geom_vline(xintercept=0) +
		geom_abline(slope=1, intercept=0, linetype="dashed") +
		geom_point(color=neonatal_outcome_colors[[neonatal_outcome_title]], size=4, alpha=0.7) +
		geom_text_repel(
			aes(label=metabolite_pair_label)) +
		ggtitle(paste0(neonatal_outcome_title, " correlation disruption"))
	#print(plt)
}
dev.off()

# Better visualization using correlation networks
corrnet_dir <- "./results/correlation_network_data/metabolites_only/"
tsne_positions <- read_csv(paste0(corrnet_dir, "tsne_positions.csv"))
nodes <- read_csv(paste0(corrnet_dir, "nodes.csv"))
nodes <- merge(nodes, metabolite_labels %>% rename(feature=raw_feature_name), by="feature") %>%
	select(feature=feature_label)
edges <- read_csv(paste0(corrnet_dir, "edges.csv"))
edges <- edges %>%
	mutate(
		metabolite_from=nodes$feature[edges$from],
		metabolite_to=nodes$feature[edges$to]
	)
create_plots <- function(corrnet_graph, tsne_positions, plot_title) {
	unlabeled_plt <- ggraph(
    corrnet_graph,
    layout=tsne_positions) +
    geom_edge_link(aes(color=rho_diff, width=abs_rho_diff), alpha=0.5) +
		scale_edge_colour_gradient2(low="#2166ac", mid="#f2f2f2", high="#b2182b") +
		scale_edge_width(range=c(0.05, 4)) +
    geom_node_point(color="dark grey", size=6, alpha=1.0) +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank(),
          axis.text.x=element_blank(), axis.text.y=element_blank(),
          axis.ticks.x=element_blank(), axis.ticks.y=element_blank(),
          aspect.ratio=1) +
    ggtitle(plot_title)
  complete_labeled_plt <- unlabeled_plt +
    geom_node_text(aes(label=feature), size=3, repel=TRUE,
                  segment.size=0.25, color="black",
                  max.iter=1000)
	return(list("unlabeled_plt"=unlabeled_plt, "complete_labeled_plt"=complete_labeled_plt))
}

disruption_results <- list()

pdf("./results/correlation_disruption_networks.pdf", width=7, height=7, useDingbats=FALSE)
for (outcome_for_network in neonatal_outcomes) {
	# Create weighted edge graph from outcome correlation network
	edge_annotation <- results[[outcome_for_network]][["corr_df"]] %>%
		separate(metabolite_pair, c("metabolite_from", "metabolite_to"), sep=",")
	merged_edges <- merge(
		edges, edge_annotation,
		by=c("metabolite_from", "metabolite_to")) %>%
		mutate(
			rho_diff=case_when(
				case_spearman_rho > 0 & control_spearman_rho < 0 ~ case_spearman_rho + control_spearman_rho,
				case_spearman_rho < 0 & control_spearman_rho > 0 ~ case_spearman_rho + control_spearman_rho,
				TRUE ~ case_spearman_rho - control_spearman_rho),
			abs_rho_diff=abs(case_spearman_rho - control_spearman_rho)) %>%
		mutate(
			direction_change=case_when(
				(case_spearman_rho > 0) & (control_spearman_rho < 0) ~ "flip",
				(case_spearman_rho < 0) & (control_spearman_rho > 0) ~ "flip",
				TRUE ~ "same"
			)
		)

	# Create graph
	edges_same_dir <- merged_edges %>% filter(direction_change != "flip")
	corrnet_graph <- tbl_graph(
			nodes=nodes,
			edges=edges_same_dir,
			directed=FALSE)
	plots <- create_plots(
			corrnet_graph, tsne_positions,
			plot_title=paste0(outcome_for_network, " disruption"))
	print(plots$unlabeled_plt)

	edges_flipped_dir <- merged_edges %>% filter(direction_change == "flip")
	corrnet_graph <- tbl_graph(
			nodes=nodes,
			edges=edges_flipped_dir,
			directed=FALSE)
	plots <- create_plots(
			corrnet_graph, tsne_positions,
			plot_title=paste0(outcome_for_network, " (flipped directionality)"))
	print(plots$unlabeled_plt)

	# Plot some examples of top flipped directions in terms of absolute
	# value
	formatted_outcome_name <- outcome_for_network %>% gsub("_any", "", .) %>% toupper()

	# Save results
	flipped_edges_df <- edges_flipped_dir %>%
		mutate(outcome=formatted_outcome_name)
	disruption_results <- append(disruption_results, list(flipped_edges_df))

	# Scatterplots for some of the top disrupted metabolite correlations
	top_five_disruptions <- edges_flipped_dir %>% slice_max(abs_rho_diff, n=5)
	metabolites_df <- bind_rows(
		results[[outcome]][["case_metabolites"]] %>% mutate(outcome_label=formatted_outcome_name),
		results[[outcome]][["control_metabolites"]] %>% mutate(outcome_label="Control")
	) %>%
	mutate(outcome_label=factor(outcome_label, levels=c("Control", formatted_outcome_name)))
	metabolites_to_plot <- mapply(
		list, top_five_disruptions$metabolite_from,
		top_five_disruptions$metabolite_to, SIMPLIFY=FALSE)
	scatterplot_colors <- c("light grey", neonatal_outcome_colors[[formatted_outcome_name]])
	names(scatterplot_colors) <-c("Control", formatted_outcome_name)
	for (metab in metabolites_to_plot) {
		plt <- ggplot(metabolites_df, aes(get(metab[[1]]), get(metab[[2]]))) +
			geom_point(aes(color=outcome_label), size=1.5, alpha=0.5) +
			scale_color_manual(values=scatterplot_colors) +
			facet_wrap(~ outcome_label, ncol=2) +
			xlab(metab[[1]]) +
			ylab(metab[[2]]) +
			theme(
				aspect.ratio=1, panel.spacing=unit(1.5, "lines"),
				strip.text=element_text(size=20), axis.title=element_text(size=20),
				plot.margin=margin(25, 25, 25, 25)) +
			guides(color=FALSE) +
			ggtitle(paste0(outcome_for_network, " metabolite values"))
		print(plt)
	}
}
dev.off()

disruption_df <- bind_rows(disruption_results) %>%
	select(-from, -to, -abs_pval_diff, -direction_change)
write_csv(disruption_df, "./results/correlation_disruption_networks_edges.csv")

