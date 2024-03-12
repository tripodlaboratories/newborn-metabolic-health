# Plot bottleneck layer output correlations with covariates
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

# Theme and Colors Setup
theme_set(theme_base(base_size=18))
model_colors <- c(
  "multi_output"="#487eb0", "large_multi_output"="#7f8fa6",
  "ensemble"="#c23616", "parallel_ensemble"="#9c88ff")
neonatal_condition_colors <- c(
  "BPD"="#4E79A7", "IVH"="#F28E2B", "NEC"="#E15759", "ROP"="#76B7B2",
  "Overlapping Conditions"="#027B8E", "None"="light grey")

# Read in bottleneck data
neonatal_conditions <- read_lines("./config/neonatal_covariates.txt")
bottle_2 <- read_csv(paste0(input_dir, "bottleneck_2_model_outputs.csv")) %>%
  mutate(condition_overlap=rowSums(.[neonatal_conditions])) %>%
  mutate(
    neonatal_outcome=case_when(
      bpd_any == 1 & condition_overlap == 1 ~ "BPD",
      ivh_any == 1 & condition_overlap == 1 ~ "IVH",
      rop_any == 1 & condition_overlap == 1 ~ "ROP",
      nec_any == 1 & condition_overlap == 1 ~ "NEC",
      condition_overlap > 1 ~ "Overlapping Conditions",
      TRUE ~ "None") %>% factor(levels=names(neonatal_condition_colors)),
    sga_any=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0),
    mortality=ifelse(`_dthind` == 1, 1, 0)
      )
bottle_5 <- read_csv(paste0(input_dir, "bottleneck_5_model_outputs.csv")) %>%
mutate(condition_overlap=rowSums(.[neonatal_conditions])) %>%
  mutate(
    neonatal_outcome=case_when(
      bpd_any == 1 & condition_overlap == 1 ~ "BPD",
      ivh_any == 1 & condition_overlap == 1 ~ "IVH",
      rop_any == 1 & condition_overlap == 1 ~ "ROP",
      nec_any == 1 & condition_overlap == 1 ~ "NEC",
      condition_overlap > 1 ~ "Overlapping Conditions",
      TRUE ~ "None") %>% factor(levels=names(neonatal_condition_colors)),
    sga_any=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0),
    mortality=ifelse(`_dthind` == 1, 1, 0)
  )
models <- bottle_2$model %>% unique()

# Plot correlations between bottleneck unit outputs and other covariates
pdf(paste0(input_dir, "correlation_analysis.pdf"), width=8.5, height=7, useDingbats=FALSE)
for (current_model in models) {
  log_info(paste("Creating bottleneck plots for:", current_model))
  model_spec <- paste(current_model, "(2 bottleneck units)")
  model_bottle_2 <- filter(bottle_2, model == current_model)

  # Plot Spearman correlation as an initial heuristic
  # bottleneck layers and covariates
  log_info("Calculating Spearman correlation for 2-unit bottleneck and covariates.")
  additional_covariates <- c("gacat", "bwtcat", "bpd_any", "ivh_any",
                             "nec_any", "rop_any", "sga_any", "mortality")
  bottleneck_colnames <- get_bottleneck_unit_columns(model_bottle_2) %>%
    select(-row_id) %>% colnames()
  spearman_vars <- c(bottleneck_colnames, additional_covariates)
  df_for_spearman <- model_bottle_2 %>%
    select(one_of(spearman_vars)) %>%
    mutate(gacat=gacat %>% as.factor() %>% as.numeric(),
           bwtcat=bwtcat %>% as.factor() %>% as.numeric())
  assertion <- assert_that(
    nrow(df_for_spearman) == nrow(model_bottle_2),
    msg="n rows for spearman df should be equal to nrows of ONE model.")
  plt <- create_spearman_plot(df_for_spearman, plot_title=paste(model_spec, "spearman"))
  print(plt)

  # Repeat for 5 unit bottleneck
  model_spec <- paste(current_model, "(5 bottleneck units)")
  model_bottle_5 <- filter(bottle_5, model == current_model)

  # Plot Spearman correlation as an initial heuristic
  # bottleneck layers and covariates
  log_info("Calculating Spearman correlation for 5-unit bottleneck and covariates.")
  bottleneck_colnames <- get_bottleneck_unit_columns(model_bottle_5) %>%
    select(-row_id) %>% colnames()
  spearman_vars <- c(bottleneck_colnames, additional_covariates)
  df_for_spearman <- model_bottle_5 %>%
    select(one_of(spearman_vars)) %>%
    mutate(gacat=gacat %>% as.factor() %>% as.numeric(),
           bwtcat=bwtcat %>% as.factor() %>% as.numeric())
  assertion <- assert_that(
    nrow(df_for_spearman) == nrow(model_bottle_5),
    msg="n rows for spearman df should be equal to nrows for ONE model")
  plt <- create_spearman_plot(df_for_spearman, plot_title=paste(model_spec, "spearman"))
  print(plt)
}
dev.off()

