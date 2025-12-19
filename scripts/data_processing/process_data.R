# Read in original data and split into maternal and neonatal conditions
suppressPackageStartupMessages({
  library(readr)
  library(tidyr)
  library(dplyr)
  library(tibble)
})

# Split main data sources into three sets, maternal, neonatal, and metadata
input_data <- read_csv("./data/raw/metabolite_cleans.csv") %>% na_if("NULL") %>%
  rowid_to_column("row_id")
neonatal_covariates <- read_lines("./config/neonatal_covariates.txt")
maternal_covariates <- read_lines("./config/maternal_covariates.txt")
metabolites <- read_lines("./config/metabolites.txt")

neonatal_conditions <- input_data %>%
  select(one_of(c("row_id", metabolites, neonatal_covariates)))
maternal_conditions <- input_data %>%
  select(one_of(c("row_id", metabolites, maternal_covariates)))

metadata <- input_data %>%
  select(row_id, everything(), -one_of(c(metabolites, maternal_covariates, neonatal_covariates)))

write_csv(neonatal_conditions, "./data/processed/neonatal_conditions.csv")
write_csv(maternal_conditions, "./data/processed/maternal_conditions.csv")
write_csv(metadata, "./data/processed/metadata.csv")

# Create additional processed data reflecting SGA, AGA, LGA:
# small, average, or large for gestational age
size_ga_cols <- c("sga_who", "sga_nichd", "aga_who", "aga_nichd", "lga_who", "lga_nichd")
size_ga <- metadata %>%
  select(row_id, one_of(size_ga_cols)) %>%
  mutate(sga_any=ifelse(sga_who == 1 | sga_nichd == 1, 1, 0),
         aga_any=ifelse(aga_who == 1 | aga_nichd == 1, 1, 0),
         lga_any=ifelse(lga_who == 1 | lga_nichd == 1, 1, 0))

write_csv(size_ga, "./data/processed/size_by_gest_age.csv")
