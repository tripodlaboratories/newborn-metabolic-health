# Plot bottleneck output correlation with metabolites
library(optparse)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(Hmisc)
library(tibble)
library(ggplot2)
library(ggcorrplot)
library(ggthemes)

source("./biobank_project/plotting/variable_correlations.R")

# Command line args
option_list <- list(
  make_option(c("-i", "--input"),
              type="character",
              help="Input directory containing bottleneck layer outputs."),
  make_option(c("-o", "--output"),
              type="character",
              help="Output file to save results.")
)
opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input
output_file <- opts$output

# Helper Functions
get_bottleneck_unit_columns <- function(input_df) {
  bottleneck_cols <- colnames(input_df)[grepl("bottleneck_unit", colnames(input_df))]
  # dplyr::distinct uses "mutate semantics", so rlang::syms needs to be used for the
  # equivalent of distinct(a, b) instead of distinct(c(a, b))
  # https://groups.google.com/forum/#!topic/manipulatr/luz2e9_QrVo
  bottleneck_data <- input_df %>% select(row_id, one_of(bottleneck_cols)) %>%
    distinct(!!!rlang::syms(bottleneck_cols), .keep_all=TRUE)
}

# Read in metabolites
neonatal_conditions <- read_lines("./config/neonatal_covariates.txt")
neonatal <- read_csv("./data/processed/neonatal_conditions.csv") %>%
	select(-one_of(neonatal_conditions))
neonatal <- neonatal[colSums(!is.na(neonatal)) > 0]

# Drop columns with large numbers of NA values
sparse_cols <- neonatal[colSums(is.na(neonatal)) > 10000] %>% colnames()
neonatal <- select(neonatal, -one_of(sparse_cols)) %>% drop_na()
metabolites <- colnames(neonatal %>% select(-row_id))

# Read in bottleneck data
bottle_2 <- read_csv(paste0(input_dir, "bottleneck_2_model_outputs.csv")) %>%
  mutate(sga_any=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0)) %>%
	merge(., neonatal, on="row_id", all.x=TRUE, all.y=FALSE)
bottle_5 <- read_csv(paste0(input_dir, "bottleneck_5_model_outputs.csv")) %>%
	mutate(sga_any=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0)) %>%
	merge(., neonatal, on="row_id", all.x=TRUE, all.y=FALSE)
models <- bottle_2$model %>% unique()

pdf(output_file, width=14, height=14, useDingbats=FALSE)
for (current_model in models) {
  model_spec <- paste(current_model, "(2 bottleneck units)")
  model_bottle_2 <- filter(bottle_2, model == current_model)

  # Plot Spearman correlation as an initial heuristic
  # bottleneck layers and covariates
  bottleneck_colnames <- get_bottleneck_unit_columns(model_bottle_2) %>%
    select(-row_id) %>% colnames()
  spearman_vars <- c(bottleneck_colnames, metabolites)
  df_for_spearman <- model_bottle_2 %>% select(one_of(spearman_vars))
  plt <- create_spearman_plot(
		df_for_spearman, plot_title=paste(model_spec, "spearman"),
		use_pvals=FALSE)
  print(plt)

  # Repeat for 5 unit bottleneck
  model_spec <- paste(current_model, "(5 bottleneck units)")
  model_bottle_5 <- filter(bottle_5, model == current_model)

	bottleneck_colnames <- get_bottleneck_unit_columns(model_bottle_5) %>%
    select(-row_id) %>% colnames()
  spearman_vars <- c(bottleneck_colnames, metabolites)
  df_for_spearman <- model_bottle_5 %>% select(one_of(spearman_vars))
  plt <- create_spearman_plot(
		df_for_spearman, plot_title=paste(model_spec, "spearman"),
		use_pvals=FALSE)
  print(plt)
}
dev.off()

