# Script for counting conditions
library(data.table)
library(dplyr)
library(readr)
library(tidyr)
library(tibble)

# Helper function
get_counts <- function(data) {
  data %>% sapply(table) %>% as.data.frame() %>%
    rownames_to_column("indicator") %>%
    gather("condition", "count", -indicator) %>%
    select(condition, everything())
}

# Read in conditions
neonatal_outcomes <- read_lines("./config/neonatal_covariates.txt")

# Read in processed data and get counts
neonatal_conditions <- fread("./data/processed/neonatal_conditions.csv")
neonatal_counts <- neonatal_conditions %>%
  select(one_of(neonatal_outcomes)) %>% get_counts()

# Get counts by condition, ethnicity, and size
metadata <- fread("./data/processed/metadata.csv")
ga_ranges <- read_lines("./config/gestational_age_ranges.txt")
meta_subset <- metadata[gacat %in% ga_ranges, ]
ga_range_ids <- meta_subset[["row_id"]]
neonatal_subset <- neonatal_conditions[row_id %in% ga_range_ids, ]
neonatal_subset[, total_conditions:=rowSums(.SD), .SDcols=neonatal_outcomes]
neonatal_subset[, any_condition:=ifelse(total_conditions >= 1, 1, 0)]
merge_cols <- c("row_id", neonatal_outcomes, "total_conditions")
neonatal_merge <- merge(neonatal_subset[, ..merge_cols], meta_subset, by="row_id")

# condition counts
condition_counts <- neonatal_merge %>% select(neonatal_outcomes) %>%
  gather("outcome", "indicator") %>% group_by(outcome, indicator) %>%
  count() %>% spread(indicator, n)

# condition counts by ethnicity
ethnicity_counts <- neonatal_merge %>% select(mrace_catm, any_condition) %>%
  gather("outcome", "indicator", -mrace_catm) %>%
  group_by(mrace_catm, outcome, indicator) %>%
  count() %>% spread(indicator, n) %>%
  mutate(case_when(
    mrace_catm == 1 ~ "Non-Hispanic White",
    mrace_catm == 2 ~ "Non-Hispanic Black",
    mrace_catm == 3 ~ "Asian",
    mrace_catm == 4 ~ "Pacific Islander",
    mrace_catm == 5 ~ "Hispanic",
    mrace_catm == 6 ~ "American Indian/Alaskan Native",
    mrace_catm == 7 ~ "Other",
    mrace_catm == 99 ~ "Missing",
    TRUE ~ "Not Listed"))


infant_sex_counts <- neonatal_merge %>% select(sex3, any_condition) %>%
  gather("outcome", "indicator", -sex3) %>%
  group_by(sex3, outcome, indicator) %>%
  count() %>% spread(indicator, n) %>%
  mutate(case_when(
    sex3 == 1 ~ "male",
    sex3 == 2 ~ "female",
    TRUE ~ "Not Listed"))
