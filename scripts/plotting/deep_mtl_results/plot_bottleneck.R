# Plot deep learning scores and other results
library(optparse)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggthemes)

theme_set(theme_base(base_size=18))

# Command line args
option_list <- list(
  make_option(c("-i", "--input"),
              type="character",
              help="Input directory containing bottleneck results."),
  make_option(c("-o", "--output"),
              type="character",
              help="Output file to save results.")
)

opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input
output_file <- opts$output

# Find which model names exist
model_dirs <- list.dirs(input_dir, full.names=FALSE, recursive=FALSE) %>%
  gsub("_bottle_.*$", "", .)
model_dirs <- model_dirs[!(model_dirs %in% c("plots", "bottleneck_layer_outputs"))]
model_names <- model_dirs %>% unique()

all_model_colors <- c(
  "multi_output"="#487eb0", "large_multi_output"="#7f8fa6",
  "ensemble"="#c23616", "parallel_ensemble"="#9c88ff",
  "additive_cov"="#43aa8b",
  "ensemble_linear"="#ea7317", "ensemble_nonlinear"="#2364aa",
  "ensemble_mini"="#73bfb8")

model_colors <- all_model_colors[names(all_model_colors) %in% model_names]

all_model_scores <- vector("list")
for (model_name in names(model_colors)) {
  score_files <- Sys.glob(
    paste0(input_dir, model_name, "*", "/scores.csv"))
  for (score_file in score_files) {
    bottleneck_n <- score_file %>% str_extract("bottle_[0-9]*") %>%
      gsub("bottle_", "", .) %>% as.numeric()
    model_scores <- read_csv(score_file, col_types=cols()) %>% select(-X1) %>%
      mutate(model=model_name,
             bottleneck_n=bottleneck_n)
    all_model_scores <- append(all_model_scores, list(model_scores))
  }
}
scores <- bind_rows(all_model_scores) %>%
  mutate(task=gsub("_any", "", task) %>% toupper(),
         model=factor(model, levels=names(model_colors)))

mean_scores <- scores %>% group_by(task, model, bottleneck_n) %>%
  summarize_all(mean) %>% select(-iter)

pdf(output_file, width=8, height=7, useDingbats=FALSE)
for (neonatal_condition in unique(mean_scores$task)) {
  condition_results <- mean_scores %>% filter(task == neonatal_condition)
  auroc_plt <- ggplot(condition_results, aes(x=bottleneck_n, y=auroc, color=model)) +
    geom_line() +
    geom_point() +
    scale_color_manual(values=model_colors) +
    ggtitle(neonatal_condition)
  aupr_plt <- ggplot(condition_results, aes(x=bottleneck_n, y=aupr, color=model)) +
    geom_line() +
    geom_point() +
    scale_color_manual(values=model_colors) +
    ggtitle(neonatal_condition)
  print(auroc_plt)
  print(aupr_plt)
}
dev.off()
