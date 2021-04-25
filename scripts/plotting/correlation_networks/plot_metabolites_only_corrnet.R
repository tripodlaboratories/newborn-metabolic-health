# Plot correlation networks for metabolites and outcomes
suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(readr)
  library(doParallel)
  library(assertthat)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(WGCNA)
  library(igraph)
  library(Rtsne)
  library(tidygraph)
  library(ggthemes)
  library(ggraph)
})

source("./scripts/plotting/correlation_networks/utils.R")

# Helper Functions
create_metabolite_color_plots <- function(correlation_network, plot_title, fill_colors=metabolite_type_colors) {
  unlabeled_plt <- ggraph(
    correlation_network$graph,
    layout=correlation_network$tsne_positions) +
    geom_edge_link(alpha=0.2, color="grey") +
    geom_node_point(
      aes(color=metabolite_type), size=6, alpha=0.8) +
    scale_color_manual(values=fill_colors) +
    labs(fill="Metabolite Type") +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank(),
          axis.text.x=element_blank(), axis.text.y=element_blank(),
          axis.ticks.x=element_blank(), axis.ticks.y=element_blank(),
          aspect.ratio=1) +
    ggtitle(plot_title)
  complete_labeled_plt <- unlabeled_plt +
    geom_node_text(aes(label=feature), size=3, repel=TRUE,
                  segment.size=0.25, color="black",
                  max.iter=1000)
  return(list("unlabeled"=unlabeled_plt, "labeled"=complete_labeled_plt))
}

create_aupr_color_plots <- function(correlation_network, plot_title) {
  unlabeled_plt <- ggraph(
    correlation_network$graph,
    layout=correlation_network$tsne_positions) +
    geom_edge_link(alpha=0.2, color="grey") +
    geom_node_point(
      aes(color=aupr, size=aupr), alpha=0.8) +
    scale_color_gradient(low="#ececec", high="#E5311C") +
    labs(color="AUPR", size="AUPR") +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank(),
          axis.text.x=element_blank(), axis.text.y=element_blank(),
          axis.ticks.x=element_blank(), axis.ticks.y=element_blank(),
          aspect.ratio=1) +
    ggtitle(plot_title)
  complete_labeled_plt <- unlabeled_plt +
    geom_node_text(aes(label=feature), size=3, repel=TRUE,
                  segment.size=0.25, color="black",
                  max.iter=1000)
  return(list("unlabeled"=unlabeled_plt, "labeled"=complete_labeled_plt))
}

# Read in metabolite labels and additional information on the metabolite level
metabolite_data <- read_csv("./config/metabolite_labels.csv") %>%
  rename(feature=raw_feature_name, metabolite_type=`category`)
metabolite_type_colors <- c(
  "amino acid"="#59a5ec",
  "short-chain acylcarnitine"="#F9C74F",
  "long-chain acylcarnitine"="#F8961E",
  "3-hydroxy long-chain acylcarnitine"="#90BE6D",
  "free carnitine"="#3f4ec1",
  "succinylacetone"="#12bbae")

# Read in metabolite AUPR
metabolite_aupr <- read_csv("./results/neonatal/metabolite_scores/neonatal_metabolite_aupr_no_overlap.csv") %>%
  rename(feature=metabolite) %>%
  gather("outcome", "aupr", -feature)
metabolite_auroc <- read_csv("./results/neonatal/metabolite_scores/neonatal_metabolite_auroc_no_overlap.csv") %>%
  rename(feature=metabolite) %>%
  gather("outcome", "auroc", -feature)

# Read in correlation network data
corrnet_dir <- "./results/correlation_network_data/metabolites_only/"
feature_labels <- read_csv(paste0(corrnet_dir, "nodes.csv"))
edges <- read_csv(paste0(corrnet_dir, "edges.csv"))
tsne_positions <- read_csv(paste0(corrnet_dir, "tsne_positions.csv"))
outcome_corr_df <- read_csv(paste0(corrnet_dir, "outcome_correlation_rho.csv"))

# Node info
relabeled_features <- merge(
  feature_labels, metabolite_data, by="feature")
assertion <- assert_that(all(feature_labels$feature %in% relabeled_features$feature))
node_info <- merge(outcome_corr_df, metabolite_aupr, by=c("feature", "outcome"))
node_info <- merge(relabeled_features, node_info, by="feature") %>%
  select(-feature) %>%
  rename(feature=feature_label)

# Plot for all neonatal outcomes
theme_set(theme_base(base_size=18))
outcomes <- metabolite_aupr$outcome %>% unique()
pdf("./results/metabolites_corrnet.pdf", width=8.5, height=7, useDingbats=FALSE)
# Create overall graph with metabolite colors
corrnet_graph <- tbl_graph(
    # Filter for outcome only to get one single set of nodes
    nodes=node_info %>% filter(outcome == "bpd_any"),
    edges=edges,
    directed=FALSE)
  corrnet <- list(graph=corrnet_graph, tsne_positions=tsne_positions)
  plots <- create_metabolite_color_plots(corrnet, plot_title="overview")
  print(plots$unlabeled)
  print(plots$labeled)

# Create graphs for individual outcomes
for(current_outcome in outcomes) {
  nodes_for_outcome <- node_info %>% filter(outcome == current_outcome)
  corrnet_graph <- tbl_graph(
    nodes=nodes_for_outcome,
    edges=edges,
    directed=FALSE)
  corrnet <- list(graph=corrnet_graph, tsne_positions=tsne_positions)
  plots <- create_aupr_color_plots(corrnet, plot_title=current_outcome)
  print(plots$unlabeled)
  print(plots$labeled)
}
dev.off()

