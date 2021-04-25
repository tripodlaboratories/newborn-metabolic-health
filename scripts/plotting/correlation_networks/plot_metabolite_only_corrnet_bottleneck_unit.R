# Plot correlation networks for metabolites and one-layer bottleneck
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
create_plots <- function(correlation_network, plot_title) {
  unlabeled_plt <- ggraph(
    correlation_network$graph,
    layout=correlation_network$tsne_positions) +
    geom_edge_link(alpha=0.2, color="grey") +
    geom_node_point(
      aes(color=rho, size=abs(rho)), alpha=0.8) +
    scale_color_distiller(type="div", palette="RdBu") +
    labs(color="Rho", size="Absolute\nCorrelation\nValue") +
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

# Read in feature data and filter on gestational age
included_ga <- read_lines("./config/gestational_age_ranges.txt")
metadata <- fread("./data/processed/metadata.csv")
gest_age <- metadata %>% select(row_id, gacat) %>% filter(gacat %in% included_ga)

neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt")
neonatal <- fread("./data/processed/neonatal_conditions.csv") %>%
  filter(row_id %in% gest_age[["row_id"]])

features <- neonatal %>% select(-one_of(neonatal_outcomes))
sparse_cols <- colnames(features)[colSums(is.na(features)) > 5000]
features <- select(features, -one_of(sparse_cols)) %>% drop_na()

# Calculate correlation with one-unit bottleneck
bottleneck <- fread("./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/valid_bottleneck.csv")
setnames(bottleneck, "V1", "row_id")
merged_df <- merge(
  features,
  bottleneck %>% group_by(row_id) %>% dplyr::summarize(bottleneck_unit_0=mean(bottleneck_unit_0)),
  by="row_id")

calculate_outcome_correlation <- function(features_df, outcome_vector) {
  outcome_corr_rho <- mclapply(
    features_df,
    function(x, y) {cor.test(x, y, method="spearman", exact=FALSE)$estimate},
    y=outcome_vector
	)
  outcome_rho <- do.call(rbind, outcome_corr_rho) %>%
    as.data.frame() %>%
    rownames_to_column("feature")
}
bottleneck_corr <- calculate_outcome_correlation(
  merged_df %>% select(-bottleneck_unit_0, -row_id), merged_df[["bottleneck_unit_0"]])

# Read in correlation network data
corrnet_dir <- "./results/correlation_network_data/metabolites_only/"
feature_labels <- read_csv(paste0(corrnet_dir, "nodes.csv"))
edges <- read_csv(paste0(corrnet_dir, "edges.csv"))
tsne_positions <- read_csv(paste0(corrnet_dir, "tsne_positions.csv"))

# Node info
relabeled_features <- merge(
  feature_labels, metabolite_data, by="feature")
assertion <- assert_that(all(feature_labels$feature %in% relabeled_features$feature))
node_info <- merge(relabeled_features, bottleneck_corr, by="feature") %>%
  select(-feature) %>%
  rename(feature=feature_label)
write_csv(node_info, "./results/metabolites_bottleneck_corr.csv")

# Plot for all neonatal outcomes
theme_set(theme_base(base_size=18))
pdf("./results/metabolites_corrnet_bottleneck.pdf", width=8.5, height=7, useDingbats=FALSE)
# Create overall graph with metabolite colors
corrnet_graph <- tbl_graph(
    # Filter for outcome only to get one single set of nodes
    nodes=node_info,
    edges=edges,
    directed=FALSE)
corrnet <- list(graph=corrnet_graph, tsne_positions=tsne_positions)
plots <- create_plots(corrnet, plot_title="corr with bottleneck")
print(plots$unlabeled)
print(plots$labeled)
dev.off()

