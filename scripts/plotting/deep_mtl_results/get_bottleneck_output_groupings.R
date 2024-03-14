# TODO: Plot bottleneck layer outputs projected into 2D space to separate samples
library(optparse)
library(logger)
library(assertthat)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(Rtsne)
library(parallelDist)
library(tibble)
library(umap)
library(ggplot2)
library(ggcorrplot)
library(ggthemes)

source("./biobank_project/plotting/variable_correlations.R")

# Set up logging
log_threshold(DEBUG)
log_layout(layout_glue_colors)

# args
option_list <- list(
  make_option(c("-i", "--input"),
              type="character",
              help="Input directory containing bottleneck results.")
)
opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input

# Helper Functions
get_bottleneck_unit_columns <- function(input_df) {
  bottleneck_cols <- colnames(input_df)[grepl("bottleneck_unit", colnames(input_df))]
  # dplyr::distinct uses "mutate semantics", so rlang::syms needs to be used for the
  # equivalent of distinct(a, b) instead of distinct(c(a, b))
  # https://groups.google.com/forum/#!topic/manipulatr/luz2e9_QrVo
  bottleneck_data <- input_df %>% select(row_id, one_of(bottleneck_cols)) %>%
    distinct(!!!rlang::syms(bottleneck_cols), .keep_all=TRUE)
}

get_tsne_coords <- function(input_df, initial_dims=50, perplexity=30, num_threads=12) {
  bottleneck_data <- get_bottleneck_unit_columns(input_df)
  tsne_results <- Rtsne(bottleneck_data %>% select(-row_id),
                        initial_dims=initial_dims,
                        perplexity=perplexity, num_threads=num_threads,
                        max_iter=3000)
  colnames(tsne_results$Y) <- c("TSNE1", "TSNE2")
  tsne_df <- bind_cols(bottleneck_data %>% select(row_id), as_tibble(tsne_results$Y)) %>%
    merge(., input_df, all.x=TRUE, all.y=FALSE, by="row_id")
}

get_umap_coords <- function(input_df, ...) {
  bottleneck_data <- get_bottleneck_unit_columns(input_df)
  umap_results <- umap(bottleneck_data %>% select(-row_id), ...)
  colnames(umap_results$layout) <- c("UMAP1", "UMAP2")
  umap_df <- bind_cols(
    bottleneck_data %>% select(row_id), as_tibble(umap_results$layout)) %>%
    merge(., input_df, all.x=TRUE, all.y=FALSE, by="row_id")
 }

create_umap_plot <- function(umap_df, plot_title, var_for_color, color_values=NULL) {
  if (!is.null(color_values)) {
    plt <- ggplot(umap_df, aes(x=UMAP1, y=UMAP2, color=get(var_for_color))) +
      geom_point(alpha=0.6) +
      scale_color_manual(values=color_values) +
      guides(color=guide_legend(title=var_for_color)) +
      ggtitle(plot_title)
  } else {
    plt <- ggplot(umap_df, aes(x=UMAP1, y=UMAP2, color=get(var_for_color))) +
      geom_point(alpha=0.6) +
      guides(color=guide_legend(title=var_for_color)) +
      ggtitle(plot_title)
  }
  return(plt)
}

# Theme and Colors Setup
theme_set(theme_base(base_size=18))
model_colors <- c(
  "multi_output"="#487eb0", "large_multi_output"="#7f8fa6",
  "ensemble"="#c23616", "parallel_ensemble"="#9c88ff")
neonatal_condition_colors <- c(
  "BPD"="#4E79A7", "IVH"="#F28E2B", "NEC"="#E15759", "ROP"="#76B7B2",
  "Overlapping"="#027B8E", "None"="light grey")

# Read in bottleneck data
set.seed(101)
neonatal_conditions <- read_lines("./config/neonatal_covariates.txt")
metadata_cols <- c(
  "sex3", "mrace_catm", "frace_catm", "lga_who", "lga_nichd", "fdeath2",
  "bmi_catm", "cig_catm", "precig_catm", "ptbsub", "ipi_cat")
metadata <- read_csv("./data/processed/metadata.csv") %>%
  select(row_id, one_of(metadata_cols)) %>%
  mutate_at(metadata_cols, factor)

input_dir <- "./results/deep_mtl/neonatal_cases_only/bottleneck/bottleneck_layer_outputs/"
bottle_5 <- read_csv(paste0(input_dir, "bottleneck_5_model_outputs.csv")) %>%
  mutate(condition_overlap=rowSums(.[neonatal_conditions])) %>%
  mutate(
    neonatal_outcome=case_when(
      bpd_any == 1 & condition_overlap == 1 ~ "BPD",
      ivh_any == 1 & condition_overlap == 1 ~ "IVH",
      rop_any == 1 & condition_overlap == 1 ~ "ROP",
      nec_any == 1 & condition_overlap == 1 ~ "NEC",
      condition_overlap > 1 ~ "Overlapping",
      TRUE ~ "None") %>% factor(levels=names(neonatal_condition_colors)),
    sga_any=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0) %>% factor(),
    mortality=ifelse(`_dthind` == 1, 1, 0) %>% factor()
  )
models <- bottle_5$model %>% unique()

# TODO: Get groupings of umap results

# bottle_dist <- model_data %>%
#   get_bottleneck_unit_columns() %>%
#   as.matrix() %>%
#   parDist(method="euclidean")

models <- c("ensemble")

# Plot correlations between bottleneck unit outputs and other covariates
additional_covariates <- c("gacat", "sga_any", "mortality", metadata_cols)
pdf(paste0(input_dir, "umap_groupings.pdf"), width=8.5, height=7, useDingbats=FALSE)
for (current_model in models) {
  log_info(paste("Creating bottleneck plots for:", current_model))

  # Reduce bottleneck outputs with t-SNE and UMAP
  # Repeat for 5 unit bottleneck
  log_info("Calculating UMAP projections for 5-unit bottleneck.")

  model_spec <- paste(current_model, "(5 bottleneck units)")
  model_data <- bottle_5 %>% filter(model == current_model)
  bottle_k <- model_data %>%
    get_bottleneck_unit_columns() %>%
    kmeans(centers=4, nstart=25)
  model_data <- model_data %>% mutate(kmeans_cluster=factor(bottle_k$cluster))
  umap_df <- get_umap_coords(model_data, random_state=101)

  plt <- create_umap_plot(
    umap_df, plot_title=paste(model_spec, "UMAP"),
    var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
  print(plt)

  plt <- create_umap_plot(
    umap_df, plot_title="KMeans on Bottleneck Outputs",
    var_for_color="kmeans_cluster")
  print(plt)

  umap_kmeans <- umap_df %>% select(UMAP1, UMAP2) %>%
    kmeans(centers=3, nstart=25)
  umap_df <- umap_df %>% mutate(umap_group=factor(umap_kmeans$cluster))
  plt <- create_umap_plot(
    umap_df, plot_title="KMeans on UMAP Outputs",
    var_for_color="umap_group")
  print(plt)
}
dev.off()
write_csv(umap_df, paste0(input_dir, "umap_groupings.csv"))

# TODO: Perform an analysis with umap groupings and all of the metadata
umap_metadata <- merge(umap_df, metadata, by="row_id", all.y=FALSE)
comparison_cols <- c(metadata_cols, "sga_any", "gacat", "bwtcat", "mortality")
pdf(paste0(input_dir, "umap_grouping_barplots.pdf"), useDingbats=FALSE)
for (col in comparison_cols) {
  plt <- ggplot(umap_metadata, aes_string(x="umap_group", fill=col)) +
    geom_bar(position="fill") +
    labs(x="UMAP Grouping", y="Proportion")
    ggtitle(paste(col))
  print(plt)
}
dev.off()

