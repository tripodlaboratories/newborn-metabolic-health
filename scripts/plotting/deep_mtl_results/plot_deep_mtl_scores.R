# Plot deep learning scores and other results
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)

theme_set(theme_base(base_size=18))

# Neonatal Model Performance
# Colors based off of https://flatuicolors.com/palette/gb
model_colors <- c(
  "multi_output"="#487eb0", "large_multi_output"="#7f8fa6",
  "ensemble"="#c23616", "parallel_ensemble"="#9c88ff")
all_model_scores <- vector("list", length=length(model_colors))
for (model_name in names(model_colors)) {
  score_file <- paste0("./results/deep_mtl/neonatal/", model_name, "/scores.csv")
  model_scores <- read_csv(score_file) %>% select(-X1) %>%
    mutate(model=model_name)
  all_model_scores[[model_name]] <- model_scores
}
scores <- bind_rows(all_model_scores) %>%
  mutate(task=gsub("_any", "", task) %>% toupper(),
         model=factor(model, levels=names(model_colors)))

# Create plots
all_plots <- vector("list")
all_plots[["neonatal_auroc"]] <- ggplot(scores, aes(x=task, y=auroc, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ggtitle("Model Performance (AUROC)")

all_plots[["neonatal_aupr"]] <- ggplot(scores, aes(x=task, y=aupr, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ggtitle("Model Performance (AUPR)")

# Plot losses
loss_colors <- c("train_loss"="#4b6584", "test_loss"="#eb3b5a")
multioutput_losses <- read_csv("./results/deep_mtl/neonatal/multi_output/losses.csv") %>% select(-X1)
losses_tall <- gather(multioutput_losses, "Loss Type", "loss", -fold, -iter, -epoch) %>%
  group_by(epoch, `Loss Type`) %>% summarize(loss=mean(loss))
all_plots[["multi_output"]] <- ggplot(losses_tall, aes(x=epoch, y=loss, color=`Loss Type`)) +
  geom_line(size=1.5) +
  scale_color_manual(values=loss_colors) +
  ggtitle("Multioutput model training")

ens_losses <- read_csv("./results/deep_mtl/neonatal/ensemble/losses.csv") %>% select(-X1)
losses_tall <- gather(ens_losses, "Loss Type", "loss", -fold, -iter, -epoch) %>%
  group_by(epoch, `Loss Type`) %>% summarize(loss=mean(loss))
all_plots[["ensemble"]] <- ggplot(losses_tall, aes(x=epoch, y=loss, color=`Loss Type`)) + 
  geom_line(size=1.5) +
  scale_color_manual(values=loss_colors) +
  ggtitle("Ensemble model training")

par_ens_losses <- read_csv("./results/deep_mtl/neonatal/parallel_ensemble/losses.csv") %>% select(-X1)
losses_tall <- gather(par_ens_losses, "Loss Type", "loss", -fold, -iter, -epoch) %>%
  group_by(epoch, `Loss Type`) %>% summarize(loss=mean(loss))
all_plots[["parallel_ensemble"]] <- ggplot(losses_tall, aes(x=epoch, y=loss, color=`Loss Type`)) +
  geom_line(size=1.5) +
  scale_color_manual(values=loss_colors) +
  ggtitle("Parallel ensemble model training")

#####################
# Maternal Outcomes #
#####################
all_model_scores <- vector("list", length=length(model_colors))
for (model_name in names(model_colors)) {
  score_file <- paste0("./results/deep_mtl/maternal/hypertension/", model_name, "/scores.csv")
  model_scores <- read_csv(score_file) %>% select(-X1) %>%
    mutate(model=model_name)
  all_model_scores[[model_name]] <- model_scores
}

task_labels <- c(
  "Hypertension\nHistory"="htn3",
  "Gestational\nHypertension"="ghtn",
  "Mild\nPreeclampsia"="mpree",
  "Severe\nPreeclampsia"="spree",
  "Preeclampsia +\nHypertension"="superimposed",
  "PROM"="prom_any")
scores <- bind_rows(all_model_scores) %>%
  mutate(task=factor(task, levels=task_labels, labels=names(task_labels)),
         model=factor(model, levels=names(model_colors)))
all_plots[["maternal_hypertension_auroc"]] <- ggplot(scores, aes(x=task, y=auroc, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  theme(axis.text.x=element_text(size=12) ,legend.position="none") +
  ggtitle("Model Performance (AUROC)")

all_plots[["maternal_hypertension_aupr"]] <- ggplot(scores, aes(x=task, y=aupr, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  theme(axis.text.x=element_text(size=12) ,legend.position="none") +
  ggtitle("Model Performance (AUPR)")

# Repeat with diabetes tasks
task_labels <- c(
  "Gestational\nDiabetes"="gdm",
  "Type 1\nDiabetes"="dmtype1",
  "Type 2\nDiabetes"="dmtype2")
all_model_scores <- vector("list", length=length(model_colors))
for (model_name in names(model_colors)) {
  score_file <- paste0("./results/deep_mtl/maternal/diabetes/", model_name, "/scores.csv")
  model_scores <- read_csv(score_file) %>% select(-X1) %>%
    mutate(model=model_name)
  all_model_scores[[model_name]] <- model_scores
}
scores <- bind_rows(all_model_scores) %>%
  mutate(task=factor(task, levels=task_labels, labels=names(task_labels)),
         model=factor(model, levels=names(model_colors)))
all_plots[["maternal_diabetes_auroc"]] <- ggplot(scores, aes(x=task, y=auroc, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ggtitle("Model Performance (AUROC)")

all_plots[["maternal_diabetes_aupr"]] <- ggplot(scores, aes(x=task, y=aupr, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ggtitle("Model Performance (AUPR)")

##################
# Stress Results #
##################
all_model_scores <- vector("list", length=length(model_colors))
for (model_name in names(model_colors)) {
  if (model_name == "parallel_ensemble") {
    # Parallel ensemble model is not currently implemented for the stress mixed 
    next()
  }
  score_file <- paste0("./results/deep_mtl/stress/", model_name, "/scores.csv")
  model_scores <- read_csv(score_file) %>% select(-X1) %>%
    mutate(model=model_name)
  all_model_scores[[model_name]] <- model_scores
}
scores <- bind_rows(all_model_scores) %>%
  mutate(model=factor(model, levels=names(model_colors)))

classification_tasks <- c(
  "preterm", "hypertension", "pree", "superimposed", "deliv_comp_mpree", "deliv_comp_spree",
  "deliv_comp_mgesthyper", "deliv_comp_chronichyper")
regression_tasks <- c("ga_delivery_days", "bmi_pre_pregnancy")
class_scores <- scores %>% filter(task %in% classification_tasks) %>%
  mutate(task=case_when(
    task == "hypertension" ~ "histhyper",
    task == "deliv_comp_mpree" ~ "mpree\n(deliv_comp\ncolumn)",
    task == "deliv_comp_spree" ~ "spree\n(deliv_comp\ncolumn)",
    task == "deliv_comp_mgesthyper" ~ "gesthyper\n(deliv_comp\ncolumn)",
    task == "deliv_comp_chronichyper" ~ "chronichyper\n(deliv_comp\ncolumn)",
    TRUE ~ task
  ))
reg_scores <- scores %>% filter(task %in% regression_tasks) %>%
  mutate(task=case_when(
    task == "bmi_pre_pregnancy" ~ "BMI\n(pre-pregnancy)",
    task == "ga_delivery_days" ~ "GA Delivery\n(days)",
    TRUE ~ task
  ))

all_plots[["stress_auroc"]] <- ggplot(class_scores, aes(x=task, y=auroc, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  theme(legend.position="none", axis.text.x=element_text(size=11)) +
  ggtitle("Model Performance (AUROC)")

all_plots[["stress_aupr"]] <- ggplot(class_scores, aes(x=task, y=aupr, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  theme(legend.position="none", axis.text.x=element_text(size=11)) +
  ggtitle("Model Performance (AUPR)")

all_plots[["stress_mae"]] <- ggplot(reg_scores, aes(x=task, y=mae, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ggtitle("Model Performance (MAE)")

all_plots[["stress_spearman_rho"]] <- ggplot(reg_scores, aes(x=task, y=spearman_rho, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ggtitle("Model Performance (Spearman Rho)")

all_plots[["stress_spearman_pval"]] <- ggplot(
  reg_scores %>% mutate(`-log10_pval`=-log10(pval)), aes(x=task, y=`-log10_pval`, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ylab("-Log10 P Value") +
  ggtitle("Model Performance (Spearman P Value)")

all_plots[["stress_r2"]] <- ggplot(reg_scores, aes(x=task, y=r2, color=model)) +
  geom_boxplot() +
  scale_color_manual(values=model_colors) +
  ggtitle(bquote("Model Performance (" ~ R^2 ~ ")"))

# Print all plots
pdf("./results/deep_mtl/plots/model_performance.pdf", width=9, height=7.5, useDingbats=FALSE)
for (plt in all_plots) {
  print(plt)
}
dev.off()
