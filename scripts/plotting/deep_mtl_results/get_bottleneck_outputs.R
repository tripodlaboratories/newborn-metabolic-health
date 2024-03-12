# Get bottleneck outputs
library(optparse)
library(readr)
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)

setDTthreads(16)

# Args
option_list <- list(
  make_option(c("-i", "--input"),
              type="character",
              help="Input directory containing bottleneck results.")
)
opts <- parse_args(OptionParser(option_list=option_list))
input_dir <- opts$input
output_dir <- paste0(input_dir, "bottleneck_layer_outputs/")
dir.create(output_dir, recursive=TRUE)

# Read in files
bottleneck_n <- c(2, 5, 10, 20, 30)
models <- list.files(input_dir)[list.files(input_dir) != "bottleneck_layer_outputs"] %>%
  gsub("_[0-9]*$", "", .)
bottleneck_files <- Sys.glob(paste0(input_dir, "*/bottleneck.csv"))
loss_files <- Sys.glob(paste0(input_dir, "*/losses.csv"))
true_val_files <- Sys.glob(paste0(input_dir, "*/true_vals.csv"))

# Attach additional metadata to results for analysis
metadata <- fread("./data/processed/metadata.csv")
metadata_cols <- c("row_id", "gacat", "bwtcat", "sga_who", "sga_nichd",
									 "_dthind")
metadata_subset <- metadata[, ..metadata_cols]

for (i in bottleneck_n) {
  matched_files <- bottleneck_files[grep(paste0("bottle_", i, "/"), bottleneck_files)]
  matched_losses <- loss_files[grep(paste0("bottle_", i, "/"), loss_files)]
  matched_truth <- true_val_files[grep(paste0("bottle_", i, "/"), true_val_files)]
  bottleneck_dfs <- list()

  for (j in seq_along(matched_files)) {
    matched_bottleneck_file <- matched_files[[j]]
    model_name <- gsub(
	    ".*/([a-z|_]*)_bottle_[0-9]*/bottleneck.csv", "\\1", matched_bottleneck_file)
    bottleneck_outputs <- fread(matched_bottleneck_file)
    setnames(bottleneck_outputs, "V1", "row_id")
    true_vals <- fread(matched_truth[[j]])
    if ("gacat" %in% colnames(true_vals) | "bwtcat" %in% colnames(true_vals)) {
      setnames(true_vals, "gacat", "gacat_model_input")
      setnames(true_vals,  "bwtcat", "bwtcat_model_input")
    }
    setnames(true_vals, "V1", "row_id")
    true_vals_metadata <- merge(
      true_vals, metadata_subset, on="row_id", all.x=TRUE, all.y=FALSE)

    #Get mean bottleneck layer outputs
    bottleneck_cols <- grep("bottleneck_unit", colnames(bottleneck_outputs))
    bottleneck_cols <- colnames(bottleneck_outputs)[bottleneck_cols]
    mean_bottleneck <- bottleneck_outputs[
      , lapply(.SD, mean), by=row_id, .SDcols=bottleneck_cols]
    mean_bottleneck <- merge(
      mean_bottleneck, true_vals_metadata[, !c("fold", "iter")] %>% subset() %>% unique(),
      by=c("row_id"))
    mean_bottleneck[["model"]] <- model_name
    bottleneck_dfs <- append(bottleneck_dfs, list(mean_bottleneck))
  }

  filename <- paste0(output_dir, "bottleneck_", i, "_model_outputs.csv")
  fwrite(bind_rows(bottleneck_dfs), filename)
}

