# Plot results for gestational age and birth weight
library(dplyr)
library(readr)
library(tidyr)
library(ggthemes)
library(ggplot2)

theme_set(theme_base(base_size=20))

get_task_specific_coefs <- function(coefs, current_task) {
  task_specific_coefs <- coefs %>%
    filter(feature != "(Intercept)") %>%
    select(one_of(c("feature", current_task, "iter"))) %>%
    group_by(feature) %>% mutate(mean_coef=mean(get(current_task)))
  
  mean_coef_cutoff <- quantile(abs(task_specific_coefs$mean_coef), .60)
  task_specific_coefs <- task_specific_coefs %>%
    select(-iter) %>%
    filter(abs(mean_coef) >= mean_coef_cutoff) %>%
    arrange(desc(mean_coef)) %>% ungroup() %>%
    mutate(feature=factor(feature, levels=unique(feature)))
}

# Read in results
# Plot coefficients
coefs <- read_csv("./results/gestage_birthweight/single_task_coefs.csv")
gacat_coef_plt <- ggplot(
  get_task_specific_coefs(coefs, "gacat_coef"),
  aes_string(x="feature", y="gacat_coef")) +
  geom_boxplot(color=solarized_pal("blue")(1)) +
  ggtitle("Gestational Age") +
  theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1))
bwtcat_coef_plt <- ggplot(
  get_task_specific_coefs(coefs, "bwtcat_coef"),
  aes_string(x="feature", y="bwtcat_coef")) +
  geom_boxplot(color=solarized_pal("red")(1)) +
  ggtitle("Birth Weight") +
  theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1))

# Read in predictions, true values, and scores
preds <- read_csv("./results/gestage_birthweight/single_task_preds.csv") %>%
  mutate(gacat=gsub("_", "-", gacat),
         bwtcat=gsub("_", "-", bwtcat))
true_vals <- read_csv("./results/gestage_birthweight/true_values.csv") %>%
  mutate(gacat=gsub("_", "-", gacat),
         bwtcat=gsub("_", "-", bwtcat))
scores <- read_csv("./results/gestage_birthweight/single_task_scores.csv")

# Set up factor labels
gacat_labels <- true_vals$gacat %>% factor() %>% levels()
gacat_breaks <- true_vals$gacat %>% factor() %>% as.numeric() %>% unique() %>% sort()
names(gacat_labels) <- gacat_breaks
bwtcat_labels <- true_vals$bwtcat %>% factor() %>% levels()
bwtcat_breaks <- true_vals$bwtcat %>% factor() %>% as.numeric() %>% unique() %>% sort()
names(bwtcat_labels) <- bwtcat_breaks

# Create plots
int_preds <- preds %>% mutate(gacat=factor(gacat) %>% as.integer()) %>%
  group_by(row_id) %>% summarize(gacat=mean(gacat))
int_true_vals <- true_vals %>% mutate(gacat=factor(gacat) %>% as.integer())
merged_preds <- merge(
  int_preds %>% rename(gacat_pred=gacat),
  int_true_vals %>% filter(iter == 1) %>% select(-iter))
spearman_results <- cor.test(merged_preds$gacat, merged_preds$gacat_pred, method="spearman", exact=FALSE)
rho <- spearman_results$estimate %>% round(4)
gacat_pred_plt <- ggplot(merged_preds, aes(x=gacat, y=gacat_pred)) +
  geom_point(alpha=0.6, color=solarized_pal("blue")(1)) +
  xlab("Gestational Age (True Value)") +
  ylab("Gestational Age (Prediction Mean)") +
  scale_x_discrete(limits=gacat_breaks, labels=gacat_labels) +
  scale_y_discrete(limits=gacat_breaks, labels=gacat_labels) +
  ggtitle(bquote("Gestational Age Predictions" ~ rho == .(rho)))

int_preds <- preds %>% mutate(bwtcat=factor(bwtcat) %>% as.integer()) %>%
  group_by(row_id) %>% summarize(bwtcat=mean(bwtcat))
int_true_vals <- true_vals %>% mutate(bwtcat=factor(bwtcat) %>% as.integer())
merged_preds <- merge(
  int_preds %>% rename(bwtcat_pred=bwtcat),
  int_true_vals %>% filter(iter == 1) %>% select(-iter))
spearman_results <- cor.test(merged_preds$bwtcat, merged_preds$bwtcat_pred, method="spearman", exact=FALSE)
rho <- spearman_results$estimate %>% round(4)

max_predicted <- merged_preds$bwtcat_pred %>% max(na.rm=TRUE)
bwtcat_breaks_y <- bwtcat_breaks[1:max_predicted]
bwtcat_labels_y <- bwtcat_labels[1:max_predicted]

bwtcat_pred_plt <- ggplot(merged_preds, aes(x=bwtcat, y=bwtcat_pred)) +
  geom_point(alpha=0.6, color=solarized_pal("red")(1)) +
  xlab("Birth Weight (True Value)") +
  ylab("Birth Weight (Prediction Mean)") +
  scale_x_discrete(
    limits=bwtcat_breaks[bwtcat_breaks %% 5 == 1],
    labels=bwtcat_labels[bwtcat_breaks %% 5 == 1]) +
  scale_y_discrete(
    limits=bwtcat_breaks_y[bwtcat_breaks_y %% 5 == 1],
    labels=bwtcat_labels_y[bwtcat_breaks_y %% 5 == 1]) +
  ggtitle(bquote("Birth Weight Predictions" ~ rho == .(rho))) +
  theme(axis.text.x=element_text(size=12, angle=90, hjust=1, vjust=0.5),
        axis.text.y=element_text(size=12))

pdf("./results/gestage_birthweight/single_task_results.pdf", height=7, width=10, useDingbats=FALSE)
print(gacat_pred_plt)
print(gacat_coef_plt)
print(bwtcat_pred_plt)
print(bwtcat_coef_plt)
dev.off()
