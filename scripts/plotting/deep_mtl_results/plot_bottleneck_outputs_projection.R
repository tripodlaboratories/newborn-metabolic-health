# Plot bottleneck layer outputs projected into 2D space to separate samples
library(optparse)
library(logger)
library(assertthat)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(Rtsne)
library(Hmisc)
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
              help="Input directory containing bottleneck results."),
  make_option(c("-o", "--output_file"),
              type="character", default="projection_analysis.pdf",
              help="Output filename for PDF."),
  make_option(c("-m", "--models"),
              type="character", default="all",
              help="comma separated list of models of interest to include"),
  make_option(c("--by_gest_age"), type="logical", action="store_true",
              default=FALSE,
              help="Split t-SNE and UMAP projections by gestational age"),
  make_option(c("--additional_covariates"), type="logical", action="store_true",
              default=FALSE,
              help="Color t-SNE and UMAP projections using additional covariates"),
  make_option(c("--interactive_debug"), type="logical", default=FALSE,
							help="adds default options for debugging interactively")
)
opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input
output_filename <- opts$output_file
if (opts$model != "all" ) {
  models_to_use <- opts$model %>% strsplit(",") %>% unlist()
} else {
  models_to_use <- "all"
}

split_by_gest_age <- opts$by_gest_age
color_by_covariates <- opts$additional_covariates
interactive_debug <- opts$interactive_debug

full_output_filename <- paste0(input_dir, output_filename)

if (isTRUE(interactive_debug)) {
  full_output_filename <- ".scratch/null.pdf"
  input_dir <- "./results/deep_mtl/neonatal_bottleneck_validation/bottleneck_layer_outputs/"
  models_to_use <- "ensemble"
  split_by_gest_age <- TRUE
}

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

create_tsne_plot <- function(tsne_df, plot_title, var_for_color,
                             color_values=NULL) {
  if (!is.null(color_values)) {
    plt <- ggplot(tsne_df, aes(x=TSNE1, y=TSNE2, color=get(var_for_color))) +
      geom_point(alpha=0.6) +
      scale_color_manual(values=color_values) +
      guides(color=guide_legend(title=var_for_color)) +
      ggtitle(plot_title)
  } else {
    plt <- ggplot(tsne_df, aes(x=TSNE1, y=TSNE2, color=get(var_for_color))) +
      geom_point(alpha=0.6) +
      guides(color=guide_legend(title=var_for_color)) +
      ggtitle(plot_title)
  }
  return(plt)
}

get_umap_coords <- function(input_df) {
  bottleneck_data <- get_bottleneck_unit_columns(input_df)
  umap_results <- umap(bottleneck_data %>% select(-row_id))
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

read_bottleneck_outputs <- function(input_dir, bottleneck_outputs_file) {
  bottleneck_df <- read_csv(paste0(input_dir, bottleneck_outputs_file))

  if (!all(neonatal_conditions %in% colnames(bottleneck_df))) {
    bottleneck_df <- merge(
      bottleneck_df,
      read_csv("./data/processed/neonatal_conditions.csv") %>%
            select(row_id, one_of(neonatal_conditions)),
      all.x=TRUE, all.y=FALSE)
  }

  bottleneck_df <- bottleneck_df %>%
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
      mortality=ifelse(`_dthind` == 1, 1, 0) %>% factor(),
      gacat=gsub("_", "-", gacat)
    )
}

# Theme and Colors Setup
theme_set(theme_base(base_size=18))
neonatal_condition_colors <- c(
  "BPD"="#4E79A7", "IVH"="#F28E2B", "NEC"="#E15759", "ROP"="#76B7B2",
  "Overlapping"="#027B8E", "None"="light grey")

# Read in bottleneck data
set.seed(101)
neonatal_conditions <- read_lines("./config/neonatal_covariates.txt")
metadata_cols <- c("sex3", "mrace_catm", "frace_catm")
metadata <- read_csv("./data/processed/metadata.csv") %>%
  select(row_id, one_of(metadata_cols)) %>%
  mutate(sex3=factor(sex3), mrace_catm=factor(mrace_catm),
         frace_catm=factor(frace_catm))

# Read in bottleneck model outputs
bottle_2 <- read_bottleneck_outputs(input_dir, "bottleneck_2_model_outputs.csv")
bottle_5 <- read_bottleneck_outputs(input_dir, "bottleneck_5_model_outputs.csv")
bottle_10 <- read_bottleneck_outputs(input_dir, "bottleneck_10_model_outputs.csv")
all_bottle <- list("2-unit"=bottle_2, "5-unit"=bottle_5, "10-unit"=bottle_10)
models <- c()
for (i in seq_along(all_bottle)) {
  models <- append(models, all_bottle[[i]][["model"]])
}
models <- models %>% unique()
if (models_to_use == "all") {
  models <- present_models
} else {
  log_info(paste("Detected the following models in files:", paste(models, collapse=" ")))
  log_info(paste("Parsed the following models to use:", paste(models_to_use, collapse=" ")))
  present_models <- models[models %in% models_to_use]
  if (length(present_models) == 0) {
    log_warn("Provided list of models does not match output data, defaulting to all models in data")
  } else {
    models <- present_models
}

# Plot correlations between bottleneck unit outputs and other covariates
additional_covariates <- c("gacat", "sga_any", "mortality", metadata_cols)
pdf(full_output_filename, width=8.5, height=7, useDingbats=FALSE)
if (isFALSE(split_by_gest_age)) {
  for (current_model in models) {
    log_info(paste("Creating bottleneck plots for:", current_model))
    model_spec <- paste(current_model, "(2 bottleneck units)")
    model_bottle_2 <- filter(bottle_2, model == current_model)
    model_bottle_2 <- merge(model_bottle_2, metadata, by="row_id", all.y=FALSE)

    # First-pass plots of bottleneck outputs as axes
    plt <- ggplot(
      model_bottle_2 %>% filter(model == current_model),
      aes(x=bottleneck_unit_0, y=bottleneck_unit_1, color=neonatal_outcome)) +
      scale_color_manual(values=neonatal_condition_colors) +
      geom_point(alpha=0.6) +
      ggtitle(model_spec)
    print(plt)

    # Reduce bottleneck outputs with t-SNE and UMAP
    log_info("Calculating t-SNE and UMAP projections for 5-unit bottleneck.")
    model_spec <- paste(current_model, "(5 bottleneck units)")
    model_bottle_5 <- filter(bottle_5, model == current_model)
    model_bottle_5 <- merge(model_bottle_5, metadata, by="row_id", all.y=FALSE)
    tsne_df <- get_tsne_coords(model_bottle_5)
    plt <- create_tsne_plot(
      tsne_df, plot_title=paste(model_spec, "TSNE"),
      var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
    print(plt)
    umap_df <- get_umap_coords(model_bottle_5)
    plt <- create_umap_plot(
      umap_df, plot_title=paste(model_spec, "UMAP"),
      var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
    print(plt)

    if (isTRUE(color_by_covariates)) {
      log_info(paste("Creating additional plots for:",
                      paste(additional_covariates, collapse=" ")))
      for (covariate in additional_covariates) {
        plt <- create_tsne_plot(tsne_df, plot_title=paste(covariate, "TSNE"),
          var_for_color=covariate)
        print(plt)
        plt <- create_umap_plot(umap_df, plot_title=paste(covariate, "UMAP"),
          var_for_color=covariate)
        print(plt)
      }
    }

    log_info("Calculating t-SNE and UMAP projections for 10-unit bottleneck.")
    model_spec <- paste(current_model, "(10 bottleneck units)")
    model_bottle_10 <- filter(bottle_10, model == current_model)
    model_bottle_10 <- merge(model_bottle_10, metadata, by="row_id", all.y=FALSE)
    tsne_df <- get_tsne_coords(model_bottle_10)
    plt <- create_tsne_plot(
      tsne_df, plot_title=paste(model_spec, "TSNE"),
      var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
    print(plt)
    umap_df <- get_umap_coords(model_bottle_10)
    plt <- create_umap_plot(
      umap_df, plot_title=paste(model_spec, "UMAP"),
      var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
    print(plt)

    if (isTRUE(color_by_covariates)) {
      log_info(paste("Creating additional plots for:",
                      paste(additional_covariates, collapse=" ")))
      for (covariate in additional_covariates) {
        plt <- create_tsne_plot(tsne_df, plot_title=paste(covariate, "TSNE"),
          var_for_color=covariate)
        print(plt)
        plt <- create_umap_plot(umap_df, plot_title=paste(covariate, "UMAP"),
          var_for_color=covariate)
        print(plt)
      }
    }
  }
} else {
  for (current_model in models) {
    log_info("Calculating t-SNE and UMAP projections for 5-unit bottleneck by gestational age.")
    model_spec <- paste(current_model, "(5 bottleneck units)")
    model_bottle_5 <- filter(bottle_5, model == current_model)
    model_bottle_5 <- merge(model_bottle_5, metadata, by="row_id", all.y=FALSE)
    gest_ages <- model_bottle_5[["gacat"]] %>% unique() %>% sort()
    for (ga in gest_ages) {
      bottle_5_in_ga <- model_bottle_5 %>% filter(gacat == ga)
       tsne_df <- get_tsne_coords(bottle_5_in_ga)
       plt <- create_tsne_plot(
        tsne_df, plot_title=paste0(model_spec, " TSNE (", ga, "weeks )"),
        var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
      print(plt)
      umap_df <- get_umap_coords(bottle_5_in_ga)
      plt <- create_umap_plot(
        umap_df, plot_title=paste0(model_spec, " UMAP (", ga, "weeks )"),
        var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
      print(plt)
    }

    log_info("Calculating t-SNE and UMAP projections for 10-unit bottleneck by gestational age.")
    model_spec <- paste(current_model, "(10 bottleneck units)")
    model_bottle_10 <- filter(bottle_10, model == current_model)
    model_bottle_10 <- merge(model_bottle_10, metadata, by="row_id", all.y=FALSE)
    gest_ages <- model_bottle_10[["gacat"]] %>% unique()
    for (ga in gest_ages) {
      bottle_10_in_ga <- model_bottle_10 %>% filter(gacat == ga)
      tsne_df <- get_tsne_coords(bottle_10_in_ga)
      plt <- create_tsne_plot(
        tsne_df, plot_title=paste0(model_spec, " TSNE (", ga, "weeks )"),
        var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
      print(plt)
      umap_df <- get_umap_coords(bottle_10_in_ga)
      plt <- create_umap_plot(
        umap_df, plot_title=paste0(model_spec, " UMAP (", ga, "weeks )"),
        var_for_color="neonatal_outcome", color_values=neonatal_condition_colors)
      print(plt)
    }
  }
}
dev.off()

