##########################################################################
## This script generates all subgroup visuals for bottleneck prediction ##
##########################################################################
# All library imports
library(ggplot2)
library(ggthemes)
library(readxl)
library(yardstick)
library(optparse)

# parse arguments
option_list <- list(
make_option(
		c("--compare_prc_vs_random"), action="store_true", default=FALSE,
		help="Add random classifier performance to precision recall curves.")
)
opts <- parse_args(OptionParser(option_list=option_list))
add_random_to_prc <- opts$compare_prc_vs_random

#some utility objects used for vis loop
outcome_order <- c("bpd", "ivh", "nec", "rop")
outcomes_colors <- tableau_color_pal("Tableau 10")(length(outcome_order))
names(outcomes_colors) <- outcome_order

# set theme size default
theme_set(theme_base(base_size=24))

###################################################
## Start of visualization loop over all outcomes ##
###################################################

#itereate over outcomes to construct kfold + val dataframes
PR_df <- data.frame()
PR_val_df <- data.frame()

#itereate over outcomes to construct kfold + val dataframes
ROC_df <- data.frame()
ROC_val_df <- data.frame()

#load baseline measurements
bottleneck_spreadsheet <- "./results/subgroup_discovery/subgroup_bottleneck_results.xlsx"
baseline_list <- as.data.frame(read_xlsx(bottleneck_spreadsheet, "baseline @ 20% Data"))
baseline_list[,2:5] <- round(baseline_list[,2:5],3)

#load baseline_rand measurements at 20%
baseline_rand_list <- as.data.frame(read_xlsx(bottleneck_spreadsheet, "rand baseline @ 20% Data"))
baseline_rand_list[,2:5] <- round(baseline_rand_list[,2:5],3)


for(outcome in outcome_order){
    PR_20 = as.data.frame(read_xlsx(bottleneck_spreadsheet, paste0(outcome,"-Kfold PR @ 20%")))
    PR_val_20 = as.data.frame(read_xlsx(bottleneck_spreadsheet, paste0(outcome,"-Val PR @ 20%")))

    PR_20 <- cbind(PR_20, rep(outcome, nrow(PR_20)))
    PR_val_20 <- cbind(PR_val_20, rep(outcome, nrow(PR_val_20)))

    PR_df <- rbind(PR_df, PR_20)
    PR_val_df <- rbind(PR_val_df, PR_val_20)

    ROC_20 = as.data.frame(read_xlsx(bottleneck_spreadsheet, paste0(outcome,"-Kfold ROC @ 20%")))
    ROC_val_20 = as.data.frame(read_xlsx(bottleneck_spreadsheet, paste0(outcome,"-Val ROC @ 20%")))

    ROC_20 <- cbind(ROC_20, rep(outcome, nrow(ROC_20)))
    ROC_val_20 <- cbind(ROC_val_20, rep(outcome, nrow(ROC_val_20)))

    ROC_df <- rbind(ROC_df, ROC_20)
    ROC_val_df <- rbind(ROC_val_df, ROC_val_20)
}
colnames(PR_df) <- c("precision", "recall", "outcome")
colnames(PR_val_df) <- c("precision", "recall", "outcome")

colnames(ROC_df) <- c("TPR", "FPR", "outcome")
colnames(ROC_val_df) <- c("TPR", "FPR", "outcome")

# Code that adds baseline prediction
if (add_random_to_prc) {
		PR_combined_plot <- ggplot(PR_df, aes(x=recall, y=precision, color=outcome))+geom_path()+
								scale_color_tableau() + ylim(0, 1) +xlab("Recall")+ylab("Precision")+ggtitle("K-Fold CV Test Set")+
		annotate("text", x=0.05, y=0.31, size=6, color=outcomes_colors[["bpd"]],
						label=paste0("BPD AUC: ",  baseline_list[baseline_list$outcome == "bpd", "kfold AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "bpd", "kfold AUPRC"] , ")"), hjust=0) +
		annotate("text", x=0.05, y=0.24, size=6, color=outcomes_colors[["ivh"]],
						label=paste0("IVH AUC: ", baseline_list[baseline_list$outcome == "ivh", "kfold AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "ivh", "kfold AUPRC"] , ")"), hjust=0) +
		annotate("text", x=0.05, y=0.17, size=6, color=outcomes_colors[["nec"]],
						label=paste0("NEC AUC: ", baseline_list[baseline_list$outcome == "nec", "kfold AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "nec", "kfold AUPRC"] , ")"), hjust=0) +
		annotate("text", x=0.05, y=0.10, size=6, color=outcomes_colors[["rop"]],
						label=paste0("ROP AUC: ", baseline_list[baseline_list$outcome == "rop", "kfold AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "rop", "kfold AUPRC"] , ")"), hjust=0) +
		labs(color="outcome") + theme(aspect.ratio=1)

		PR_combined_val_plot <- ggplot(PR_val_df, aes(x=recall, y=precision, color=outcome))+geom_path()+
								scale_color_tableau() +ylim(0, 1) + xlab("Recall") + ylab("Precision") +ggtitle("Holdout Validation Set") +
		annotate("text", x=0.05, y=0.31, size=6, color=outcomes_colors[["bpd"]],
						label=paste0("BPD AUC: ",  baseline_list[baseline_list$outcome == "bpd", "val AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "bpd", "val AUPRC"] , ")"), hjust=0) +
		annotate("text", x=0.05, y=0.24, size=6, color=outcomes_colors[["ivh"]],
						label=paste0("IVH AUC: ", baseline_list[baseline_list$outcome == "ivh", "val AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "ivh", "val AUPRC"] , ")"), hjust=0) +
		annotate("text", x=0.05, y=0.17, size=6, color=outcomes_colors[["nec"]],
						label=paste0("NEC AUC: ", baseline_list[baseline_list$outcome == "nec", "val AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "nec", "val AUPRC"] , ")"), hjust=0) +
		annotate("text", x=0.05, y=0.10, size=6, color=outcomes_colors[["rop"]],
						label=paste0("ROP AUC: ", baseline_list[baseline_list$outcome == "rop", "val AUPRC"], " (Baseline: ", baseline_rand_list[baseline_rand_list$outcome == "rop", "val AUPRC"] , ")"), hjust=0) +
		labs(color="outcome") + theme(aspect.ratio=1)
} else {
PR_combined_plot <- ggplot(PR_df, aes(x=recall, y=precision, color=outcome))+geom_path()+
              scale_color_tableau() + ylim(0, 1) +xlab("Recall")+ylab("Precision")+ggtitle("K-Fold CV Test Set")+
	annotate("text", x=0.5, y=0.31, size=6, color=outcomes_colors[["bpd"]],
					 label=paste0("BPD AUC: ",  baseline_list[baseline_list$outcome == "bpd", "kfold AUPRC"]), hjust=0) +
	annotate("text", x=0.5, y=0.24, size=6, color=outcomes_colors[["ivh"]],
					 label=paste0("IVH AUC: ", baseline_list[baseline_list$outcome == "ivh", "kfold AUPRC"]), hjust=0) +
	annotate("text", x=0.5, y=0.17, size=6, color=outcomes_colors[["nec"]],
					 label=paste0("NEC AUC: ", baseline_list[baseline_list$outcome == "nec", "kfold AUPRC"]), hjust=0) +
	annotate("text", x=0.5, y=0.10, size=6, color=outcomes_colors[["rop"]],
					 label=paste0("ROP AUC: ", baseline_list[baseline_list$outcome == "rop", "kfold AUPRC"]), hjust=0) +
	labs(color="outcome") + theme(aspect.ratio=1)

PR_combined_val_plot <- ggplot(PR_val_df, aes(x=recall, y=precision, color=outcome))+geom_path()+
              scale_color_tableau() +ylim(0, 1) + xlab("Recall") + ylab("Precision") +ggtitle("Holdout Validation Set") +
	annotate("text", x=0.5, y=0.31, size=6, color=outcomes_colors[["bpd"]],
					 label=paste0("BPD AUC: ",  baseline_list[baseline_list$outcome == "bpd", "val AUPRC"]), hjust=0) +
	annotate("text", x=0.5, y=0.24, size=6, color=outcomes_colors[["ivh"]],
					 label=paste0("IVH AUC: ", baseline_list[baseline_list$outcome == "ivh", "val AUPRC"]), hjust=0) +
	annotate("text", x=0.5, y=0.17, size=6, color=outcomes_colors[["nec"]],
					 label=paste0("NEC AUC: ", baseline_list[baseline_list$outcome == "nec", "val AUPRC"]), hjust=0) +
	annotate("text", x=0.5, y=0.10, size=6, color=outcomes_colors[["rop"]],
					 label=paste0("ROP AUC: ", baseline_list[baseline_list$outcome == "rop", "val AUPRC"]), hjust=0) +
	labs(color="outcome") + theme(aspect.ratio=1)
}

pdf("./results/subgroup_discovery/bottleneck_kfold_PR_curves.pdf", height=7, width=7, useDingbats=FALSE)
print(PR_combined_plot)
dev.off()

pdf("./results/subgroup_discovery/bottleneck_val_PR_curves.pdf", height=7, width=7, useDingbats=FALSE)
print(PR_combined_val_plot)
dev.off()


ROC_combined_plot <- ggplot(ROC_df, aes(x=FPR, y=TPR, color=outcome))+geom_path()+
              scale_color_tableau() +
	geom_segment(aes(x=0, xend=1, y=0, yend=1), color="grey", linetype="dashed")+ ylim(0, 1) +xlab("FPR")+ylab("TPR")+ggtitle("K-Fold CV Test Set")+
	annotate("text", x=0.5, y=0.31, size=6, color=outcomes_colors[["bpd"]],
					 label=paste0("BPD AUC: ",  baseline_list[baseline_list$outcome == "bpd", "kfold AUROC"]), hjust=0) +
	annotate("text", x=0.5, y=0.24, size=6, color=outcomes_colors[["ivh"]],
					 label=paste0("IVH AUC: ", baseline_list[baseline_list$outcome == "ivh", "kfold AUROC"]), hjust=0) +
	annotate("text", x=0.5, y=0.17, size=6, color=outcomes_colors[["nec"]],
					 label=paste0("NEC AUC: ", baseline_list[baseline_list$outcome == "nec", "kfold AUROC"]), hjust=0) +
	annotate("text", x=0.5, y=0.1, size=6, color=outcomes_colors[["rop"]],
					 label=paste0("ROP AUC: ", baseline_list[baseline_list$outcome == "rop", "kfold AUROC"]), hjust=0) +
	labs(color="outcome") + theme(aspect.ratio=1)

ROC_combined_val_plot <- ggplot(ROC_val_df, aes(x=FPR, y=TPR, color=outcome))+geom_path()+
              scale_color_tableau() +
	geom_segment(aes(x=0, xend=1, y=0, yend=1), color="grey", linetype="dashed")+ ylim(0, 1) + xlab("FPR") + ylab("TPR") +ggtitle("Holdout Validation Set") +
	annotate("text", x=0.5, y=0.31, size=6, color=outcomes_colors[["bpd"]],
					 label=paste0("BPD AUC: ",  baseline_list[baseline_list$outcome == "bpd", "val AUROC"]), hjust=0) +
	annotate("text", x=0.5, y=0.24, size=6, color=outcomes_colors[["ivh"]],
					 label=paste0("IVH AUC: ", baseline_list[baseline_list$outcome == "ivh", "val AUROC"]), hjust=0) +
	annotate("text", x=0.5, y=0.17, size=6, color=outcomes_colors[["nec"]],
					 label=paste0("NEC AUC: ", baseline_list[baseline_list$outcome == "nec", "val AUROC"]), hjust=0) +
	annotate("text", x=0.5, y=0.1, size=6, color=outcomes_colors[["rop"]],
					 label=paste0("ROP AUC: ", baseline_list[baseline_list$outcome == "rop", "val AUROC"]), hjust=0) +
	labs(color="outcome") + theme(aspect.ratio=1)

pdf("./results/subgroup_discovery/bottleneck_kfold_ROC_curves.pdf", height=7, width=7, useDingbats=FALSE)
print(ROC_combined_plot)
dev.off()

pdf("./results/subgroup_discovery/bottleneck_val_ROC_curves.pdf", height=7, width=7, useDingbats=FALSE)
print(ROC_combined_val_plot)
dev.off()
