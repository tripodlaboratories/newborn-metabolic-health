# Create Venn Diagram of Overlapping Outcomes
suppressPackageStartupMessages({
  library(readr)
  library(tidyr)
  library(dplyr)
  library(eulerr)
  library(RColorBrewer)
  library(ggthemes)
  library(ggplot2)
})

# Read in data
neonatal_tasks <- read_lines("./config/neonatal_covariates.txt")
neonatal_conditions <- read_csv("./data/processed/neonatal_conditions.csv") %>%
  select(row_id, one_of(neonatal_tasks)) %>%
  rename(NEC=nec_any, BPD=bpd_any, IVH=ivh_any, ROP=rop_any)

# Create plot of neonatal conditions for maternal conditions
maternal_tasks <- read_lines("./config/maternal_covariates.txt")
maternal_conditions <- read_csv("./data/processed/maternal_conditions.csv") %>%
  select(row_id, one_of(maternal_tasks) %>% sort())

merged_conditions <- merge(neonatal_conditions, maternal_conditions) %>%
  drop_na() %>%
  mutate(neonatal_overlap=rowSums(.[c("NEC", "BPD", "IVH", "ROP")]),
         maternal_overlap=rowSums(.[maternal_tasks])) %>%
  mutate(
    maternal_condition=case_when(
      gdm == 1 & maternal_overlap == 1~ "Gestational\nDiabetes",
      ghtn == 1 & maternal_overlap == 1 ~ "Gestational\nHypertension",
      mpree == 1 & maternal_overlap == 1 ~ "Mild\nPreeclampsia",
      spree == 1 & maternal_overlap == 1 ~ "Severe\nPreeclampsia",
      superimposed == 1 & maternal_overlap == 1 ~ "Superimposed\nPreeclampsia",
      prom_any == 1 & maternal_overlap == 1 ~ "PROM",
      maternal_overlap > 1 ~ "Overlapping\nConditions",
      TRUE ~ "None of\nthe Above"),
    neonatal_condition=case_when(
      NEC == 1 & neonatal_overlap == 1 ~ "NEC",
      ROP == 1 & neonatal_overlap == 1 ~ "ROP",
      BPD == 1 & neonatal_overlap == 1 ~ "BPD",
      IVH == 1 & neonatal_overlap == 1 ~ "IVH",
      neonatal_overlap > 1 ~ "Overlapping Conditions",
      TRUE ~ "None of the Above")) %>%
  mutate(
    maternal_condition=factor(
      maternal_condition,
      levels=c("Gestational\nDiabetes", "Gestational\nHypertension",
               "Mild\nPreeclampsia", "Severe\nPreeclampsia", "Superimposed\nPreeclampsia",
               "PROM", "Overlapping\nConditions", "None of\nthe Above")),
    neonatal_condition=factor(
      neonatal_condition,
      levels=c("BPD", "IVH", "NEC", "ROP", "Overlapping Conditions", "None of the Above")))

# Get data for size for gestational age
size_ga_labels <- c("sga_any"="SGA", "aga_any"="AGA", "lga_any"="LGA")
size_ga <- read_csv("./data/processed/size_by_gest_age.csv") %>%
  select(row_id, one_of(names(size_ga_labels))) %>%
  mutate(size_category=case_when(sga_any == 1 ~ "SGA",
                                 lga_any == 1 ~ "LGA",
                                 aga_any == 1 ~ "AGA",
                                 TRUE ~ "unknown"))
conditions_by_size <- merge(size_ga, merged_conditions)

metadata <- read_csv("./data/processed/metadata.csv") 
gest_age <- metadata %>% select(row_id, gacat)
preterm_conditions_by_size <- merge(conditions_by_size, gest_age) %>%
  filter(gacat %in% read_lines("./config/gestational_age_ranges.txt")) %>%
  filter(size_category != "unknown") %>%
  mutate(size_category=factor(size_category, levels=c("SGA", "AGA", "LGA")))

# Get data for mortality
mortality_data <- metadata %>% select(row_id, `_dthind`) %>%
  rename(mortality=`_dthind`) %>%
  mutate(mortality=case_when(
    mortality == 1 ~ "Neonatal\nDeath",
    mortality == 0 ~ "Survived",
    mortality == 2 ~ "Postneonatal\nDeath",
    TRUE ~ "Unknown"))
size_mortality <- merge(preterm_conditions_by_size, mortality_data, all.x=TRUE)

# Add a plot for neonatal overlapping conditions with gestational age
neonatal_gest_age <- merge(merged_conditions, gest_age)

# Create Plots
set.seed(0)
pdf("./results/exploratory/neonatal_venn.pdf", height=7, width=7, useDingbats=FALSE)
neonatal_conditions_only <- neonatal_conditions %>%
  filter(row_id %in% preterm_conditions_by_size[["row_id"]]) %>%
  select(sort(colnames(neonatal_conditions %>% select(-row_id))))
plot(venn(neonatal_conditions_only), quantities=TRUE, 
     fills=tableau_color_pal()(length(colnames(neonatal_conditions_only))))
dev.off()

theme_set(theme_base(base_size=18))
neonatal_condition_colors <- c(
  "BPD"="#4E79A7", "IVH"="#F28E2B", "NEC"="#E15759", "ROP"="#76B7B2",
  "Overlapping Conditions"="#027B8E", "None of the Above"="light grey")

plt <- ggplot(merged_conditions, aes(x=maternal_condition)) + 
  geom_bar(aes(fill=neonatal_condition)) + 
  scale_fill_manual(values=neonatal_condition_colors) +
  xlab("Maternal Condition Category") +
  theme(legend.position=c(0.15, 0.75), axis.text.x=element_text(size=12)) +
  ggtitle("Maternal/Neonatal Condition Overlap")

neonatal_gest_age_plt <- ggplot(
  neonatal_gest_age %>% mutate(gacat:=gsub("_", "-", gacat)),
  aes(x=gacat)) +
  geom_bar(aes(fill=neonatal_condition)) +
  scale_fill_manual(values=neonatal_condition_colors) +
  xlab("Gestational Age Range\n(weeks)") +
  theme(legend.position="none") +
  ggtitle("Neonatal Condition Overlap by Gestational Age")

# Subset on gestational age ranges
gest_age_subset <- read_lines("./config/gestational_age_ranges.txt")
neonatal_gest_subset_plt <- ggplot(
  neonatal_gest_age %>% filter(gacat %in% gest_age_subset) %>%
    mutate(gacat:=gsub("_", "-", gacat)),
  aes(x=gacat)) +
  geom_bar(aes(fill=neonatal_condition)) +
  scale_fill_manual(values=neonatal_condition_colors) +
  xlab("Gestational Age Range (weeks)") +
  theme(axis.text.x=element_text(size=20),
        axis.text.y=element_text(size=20),
        axis.title=element_text(size=20), legend.position="none") +
  ggtitle("Neonatal Condition Overlap by Gestational Age")

preterm_size_plt <- ggplot(
  preterm_conditions_by_size %>% filter(size_category != "unknown") %>%
      mutate(size_category=factor(size_category, levels=c("SGA", "AGA", "LGA"))), aes(x=size_category)) +
  geom_bar(aes(fill=neonatal_condition)) + 
  scale_fill_manual(values=neonatal_condition_colors) +
  xlab("Size for Gestational Age") +
  ggtitle("Neonatal Condition Distribution Across Size (Preterm Births)")

mortality_colors <- c("Neonatal\nDeath"="#d72c16", "Survived"="light grey",
                      "Postneonatal\nDeath"="#a10115", "Unknown"="#f1f3ce")
size_mortality_plt <- ggplot(size_mortality, aes(x=size_category)) +
  geom_bar(aes(fill=mortality)) +
  scale_fill_manual(values=mortality_colors) +
  xlab("Size for Gestational Age") +
  ggtitle("Neonatal Condition Distribution Across Size (Preterm Births)")

pdf("./results/exploratory/neonatal_overlap.pdf", width=10, height=6, useDingbats=FALSE)
print(plt)
print(neonatal_gest_age_plt)
print(neonatal_gest_subset_plt)
print(preterm_size_plt)
print(size_mortality_plt)
dev.off()
