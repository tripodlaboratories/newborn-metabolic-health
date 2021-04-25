# Utility functions for correlation networks
suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(readr)
  library(doParallel)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(WGCNA)
  library(igraph)
  library(Rtsne)
  library(tidygraph)
  library(ggthemes)
})

process_features <- function(features_file) {
  # Read in features
  input_data <- read_csv(features_file, col_types=cols()) %>% na_if("NULL") %>%
    mutate_if(is.character, as.double)
  input_data <- input_data[colSums(!is.na(input_data)) > 0]
  
  # Drop columns with large numbers of NA values
  sparse_cols <- input_data[colSums(is.na(input_data)) > 10000] %>% colnames()
  input_data <- select(input_data, -one_of(sparse_cols)) %>% drop_na()
}

create_correlation_matrices <- function(features, coefs, num_threads=0) {
  # Create a correlation matrix of features
  feature_matrix <- features %>% select(one_of(coefs$feature)) %>% as.matrix()
  feature_matrix_dims <- dim(feature_matrix)
  num_samples <- feature_matrix_dims[[1]]
  num_features <- feature_matrix_dims[[2]]
  
  corr_results <- WGCNA::cor(
    features %>% select(one_of(coefs$feature)) %>% as.matrix(),
    use="pairwise.complete.obs", method="pearson", nThreads=num_threads)
  corr_p_vals <- corPvalueFisher(
    corr_results, nSamples=num_samples)
  corr_p_vals <- corr_p_vals * num_features
  corr_p_vals[corr_p_vals > 1.0] <- 1.0
  
  # Return absolute values of correlation coefficients
  # For the matrix of p values, Boolean matrix * 1
  # is a hack to get a numerical matrix of 1s and 0s
  # corresponding to p values that are significant
  return(list(corr_mat=corr_results %>% abs(),
              pval_mat=(corr_p_vals < 0.05) * 1))
}

calculate_covariate_correlation <- function(features_df, coefs, feature_names, outcome) {
  
  merged_df <- features_df %>% select(one_of(c(feature_names, outcome)))
  outcome_vector <- merged_df[[outcome]]
  outcome_corr_rho <- mclapply(
    merged_df %>% select(-one_of(c(outcome))),
    function(x, y) {cor.test(x, y, method="spearman", exact=FALSE)$estimate},
    y=outcome_vector)
  outcome_rho <- do.call(rbind, outcome_corr_rho) %>%
    as.data.frame() %>%
    rownames_to_column("feature")
  
  overall_feature_summary <- merge(
    coefs %>% select(feature, mean_coef),
    outcome_rho, by.x="feature", by.y="feature")
  
  percentile_ranks <- c(.75, .90, .95, .975, .99, .995)
  for (percentile in percentile_ranks) {
    feature_summary <- overall_feature_summary %>% mutate(top_feature_label=ifelse(
      (abs(rho) >= quantile(abs(rho), percentile)) & (abs(mean_coef) >= quantile(abs(mean_coef), percentile)),
      feature, NA))
    num_labeled_features <- length(feature_summary$top_feature_label[!is.na(feature_summary$top_feature_label)])
    if (num_labeled_features <= 20) {
      break
    }
  }
  
  # Handle the case where there are still many features after percentile filtering
  if (num_labeled_features > 20) {
    top_ranked_features <- feature_summary %>% select(feature, mean_coef) %>% filter(mean_coef != 0) %>%
      mutate(abs_coef=abs(mean_coef)) %>% top_n(25, abs_coef)
    feature_summary <- feature_summary %>% mutate(top_feature_label=ifelse(
      feature %in% top_ranked_features$feature, feature, NA))
  }
  list(overall_feature_summary=overall_feature_summary,
       feature_summary=feature_summary)
}


get_tsne_coords <- function(
  correlation_matrix, perplexity=15, max_iter=3000, num_threads=1) {
  corr_mat_tsne <- Rtsne(
    correlation_matrix, perplexity=perplexity,
    max_iter=max_iter, num_threads=num_threads, check_duplicates=FALSE)
  tsne_positions <- corr_mat_tsne$Y
  rownames(tsne_positions) <- colnames(correlation_matrix)
  tsne_positions <- tsne_positions %>% as.data.frame() %>% select(x=V1, y=V2)
}

save_plots <- function(plots) {
  # Expect plots to be a named list
  foreach(k=seq_along(plots)) %dopar% {
    ggplot2::theme_set(ggthemes::theme_few(base_size=18))
    filename <- names(plots)[[k]]
    plt <- plots[[k]]
    if (grepl("top_labels", filename)) {
      suppressWarnings(
        ggplot2::ggsave(filename, plt, "png", width=8.75,
                        height=6, unit="in", dpi=1200)
      )
    } else {
      ggplot2::ggsave(filename, plt, "png", width=8.75,
                      height=6, unit="in", dpi=1200)
    }
    rm(plt)
    NULL
  }
}
