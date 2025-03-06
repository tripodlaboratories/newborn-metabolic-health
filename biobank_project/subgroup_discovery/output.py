import xlsxwriter


def write_excel(
        file: str,
        all_results,
        iter_results,
        pred_type: str,
        outcome_order: list):
    """
    Args:
        file: xlsx file, ccould be dynamically generated like `output_dir + metric + "_bottleneck_results.xlsx"`
        all_results: subgroup discovery results for predictions summarized across all iterations
        iter_results: subgroup discovery results for individual iterations
        outcome_order: order of outcomes to create for each worksheet, e.g., ['bpd_any', 'rop_any', 'ivh_any', 'nec_any']
    """
    workbook = xlsxwriter.Workbook(file)
    worksheet_baseline = workbook.add_worksheet("baseline")
    worksheet_baseline_mean = workbook.add_worksheet("Mean+SD Across Preds")
    worksheet_baseline_rand = workbook.add_worksheet("rand baseline")
    worksheet_20 = workbook.add_worksheet("baseline @ 20% Data")
    worksheet_20_mean = workbook.add_worksheet("Mean+SD Across Preds @ 20% Data")
    worksheet_20_rand = workbook.add_worksheet("rand baseline @ 20% Data")
    #
    for outcome in outcome_order:
        targ=outcome
        pred_type = pred_type
        #
        worksheet_train = workbook.add_worksheet(outcome+"-train")
        worksheet_val = workbook.add_worksheet(outcome+"-val")
        worksheet_pr = workbook.add_worksheet(outcome+"-Kfold PR @ 20%")
        worksheet_roc = workbook.add_worksheet(outcome+"-Kfold ROC @ 20%")
        worksheet_pr_val = workbook.add_worksheet(outcome+"-Val PR @ 20%")
        worksheet_roc_val = workbook.add_worksheet(outcome+"-Val ROC @ 20%")
        #
        subgroup_results = all_results[targ+pred_type][0]
        subgroup_val_results = all_results[targ+pred_type][1]
        #
        kfold_AUROC = all_results[targ+pred_type][2]
        kfold_AUPRC = all_results[targ+pred_type][3]
        #
        val_AUROC = all_results[targ+pred_type][4]
        val_AUPRC = all_results[targ+pred_type][5]
        #
        (precision_20, recall_20, thresholds_20) = all_results[targ+pred_type][6]
        (precision_val_20, recall_val_20, thresholds_val_20) = all_results[targ+pred_type][7]
        #
        kfold_20_AUPRC = all_results[targ+pred_type][8]
        val_20_AUPRC = all_results[targ+pred_type][9]
        #
        (fpr_20, tpr_20, thresholds_20) = all_results[targ+pred_type][10]
        (fpr_val_20, tpr_val_20, thresholds_val_20) = all_results[targ+pred_type][11]
        #
        kfold_20_AUROC = all_results[targ+pred_type][12]
        val_20_AUROC = all_results[targ+pred_type][13]
        #
        #
        worksheet_baseline.write(0,0, "outcome")
        worksheet_baseline.write(0,1, "kfold AUROC")
        worksheet_baseline.write(0,2, "kfold AUPRC")
        worksheet_baseline.write(0,3, "val AUROC")
        worksheet_baseline.write(0,4, "val AUPRC")
        #
        worksheet_baseline.write(outcome_order.index(outcome)+1, 0, outcome)
        worksheet_baseline.write(outcome_order.index(outcome)+1, 1, kfold_AUROC)
        worksheet_baseline.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC)
        worksheet_baseline.write(outcome_order.index(outcome)+1, 3, val_AUROC)
        worksheet_baseline.write(outcome_order.index(outcome)+1, 4, val_AUPRC)
        #
        #
        worksheet_20.write(0,0, "outcome")
        worksheet_20.write(0,1, "kfold AUPRC")
        worksheet_20.write(0,2, "val AUPRC")
        worksheet_20.write(0,3, "kfold AUROC")
        worksheet_20.write(0,4, "val AUROC")
        #
        worksheet_20.write(outcome_order.index(outcome)+1, 0, outcome)
        worksheet_20.write(outcome_order.index(outcome)+1, 1, kfold_20_AUPRC)
        worksheet_20.write(outcome_order.index(outcome)+1, 2, val_20_AUPRC)
        worksheet_20.write(outcome_order.index(outcome)+1, 3, kfold_20_AUROC)
        worksheet_20.write(outcome_order.index(outcome)+1, 4, val_20_AUROC)
        #
        #
        kfold_AUROC_mean = iter_results[targ+pred_type][0]
        kfold_AUROC_sd = iter_results[targ+pred_type][1]
        val_AUROC_mean = iter_results[targ+pred_type][2]
        val_AUROC_sd = iter_results[targ+pred_type][3]
        #
        kfold_AUROC_mean_20 = iter_results[targ+pred_type][4]
        kfold_AUROC_sd_20 = iter_results[targ+pred_type][5]
        val_AUROC_mean_20 = iter_results[targ+pred_type][6]
        val_AUROC_sd_20 = iter_results[targ+pred_type][7]
        #
        kfold_AUPRC_mean = iter_results[targ+pred_type][8]
        kfold_AUPRC_sd = iter_results[targ+pred_type][9]
        val_AUPRC_mean = iter_results[targ+pred_type][10]
        val_AUPRC_sd = iter_results[targ+pred_type][11]
        #
        kfold_AUPRC_mean_20 = iter_results[targ+pred_type][12]
        kfold_AUPRC_sd_20 = iter_results[targ+pred_type][13]
        val_AUPRC_mean_20 = iter_results[targ+pred_type][14]
        val_AUPRC_sd_20 = iter_results[targ+pred_type][15]
        #
        worksheet_20_mean.write(0,0, "outcome")
        worksheet_20_mean.write(0,1, "kfold AUPRC")
        worksheet_20_mean.write(0,2, "val AUPRC")
        worksheet_20_mean.write(0,3, "kfold AUROC")
        worksheet_20_mean.write(0,4, "val AUROC")
        #
        worksheet_20_mean.write(0,6, "kfold AUPRC SD")
        worksheet_20_mean.write(0,7, "val AUPRC SD")
        worksheet_20_mean.write(0,8, "kfold AUROC SD")
        worksheet_20_mean.write(0,9, "val AUROC SD")
        #
        #
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 0, outcome)
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUPRC_mean_20)
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 2, val_AUPRC_mean_20)
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 3, kfold_AUROC_mean_20)
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 4, val_AUROC_mean_20)
        #
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUPRC_sd_20)
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 7, val_AUPRC_sd_20)
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 8, kfold_AUROC_sd_20)
        worksheet_20_mean.write(outcome_order.index(outcome)+1, 9, val_AUROC_sd_20)
        #
        worksheet_baseline_mean.write(0,0, "outcome")
        worksheet_baseline_mean.write(0,1, "kfold AUROC")
        worksheet_baseline_mean.write(0,2, "kfold AUPRC")
        worksheet_baseline_mean.write(0,3, "val AUROC")
        worksheet_baseline_mean.write(0,4, "val AUPRC")
        #
        worksheet_baseline_mean.write(0,6, "kfold AUPRC SD")
        worksheet_baseline_mean.write(0,7, "val AUPRC SD")
        worksheet_baseline_mean.write(0,8, "kfold AUROC SD")
        worksheet_baseline_mean.write(0,9, "val AUROC SD")
        #
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 0, outcome)
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUROC_mean)
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC_mean)
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 3, val_AUROC_mean)
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 4, val_AUPRC_mean)
        #
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUPRC_sd)
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 7, val_AUPRC_sd)
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 8, kfold_AUROC_sd)
        worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 9, val_AUROC_sd)
        #
        #
        rand_AUROC = all_results[targ+pred_type][14]
        rand_AUPRC = all_results[targ+pred_type][15]
        #
        rand_val_AUROC = all_results[targ+pred_type][16]
        rand_val_AUPRC = all_results[targ+pred_type][17]
        #
        #
        rand_AUROC_20 = all_results[targ+pred_type][18]
        rand_AUPRC_20 = all_results[targ+pred_type][19]
        #
        rand_val_AUROC_20 = all_results[targ+pred_type][20]
        rand_val_AUPRC_20 = all_results[targ+pred_type][21]
        #
        worksheet_20_rand.write(0,0, "outcome")
        worksheet_20_rand.write(0,1, "kfold AUPRC")
        worksheet_20_rand.write(0,2, "val AUPRC")
        worksheet_20_rand.write(0,3, "kfold AUROC")
        worksheet_20_rand.write(0,4, "val AUROC")
        #
        worksheet_20_rand.write(outcome_order.index(outcome)+1, 0, outcome)
        worksheet_20_rand.write(outcome_order.index(outcome)+1, 1, rand_AUPRC_20)
        worksheet_20_rand.write(outcome_order.index(outcome)+1, 2, rand_val_AUPRC_20)
        worksheet_20_rand.write(outcome_order.index(outcome)+1, 3, rand_AUROC_20)
        worksheet_20_rand.write(outcome_order.index(outcome)+1, 4, rand_val_AUROC_20)
        #
        worksheet_baseline_rand.write(0,0, "outcome")
        worksheet_baseline_rand.write(0,1, "kfold AUROC")
        worksheet_baseline_rand.write(0,2, "kfold AUPRC")
        worksheet_baseline_rand.write(0,3, "val AUROC")
        worksheet_baseline_rand.write(0,4, "val AUPRC")
        #
        worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 0, outcome)
        worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 1, rand_AUROC)
        worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 2, rand_AUPRC)
        worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 3, rand_val_AUROC)
        worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 4, rand_val_AUPRC)
        #
        #adding column titles
        col_num=0
        row_num=0
        for col in subgroup_results.columns:
            temp = worksheet_train.write(row_num, col_num, col)
            col_num = col_num + 1
        #
        #adding subgroup result vectors
        col_num=0
        for col in subgroup_results.columns:
            row_num = 1
            for val in subgroup_results[col]:
                if(val != val): #nan check
                    temp = worksheet_train.write(row_num, col_num, "nan")
                else:
                    temp = worksheet_train.write(row_num, col_num, val)
                row_num = row_num + 1
            col_num = col_num + 1
        #
        #adding column titles
        col_num=0
        row_num=0
        for col in subgroup_val_results.columns:
            temp = worksheet_val.write(row_num, col_num, col)
            col_num = col_num + 1
        #
        #adding subgroup result vectors
        col_num=0
        for col in subgroup_results.columns:
            row_num = 1
            for val in subgroup_val_results[col]:
                if(val != val):
                    temp = worksheet_val.write(row_num, col_num, "nan")
                else:
                    temp = worksheet_val.write(row_num, col_num, val)
                row_num  = row_num + 1
            col_num = col_num + 1
        #
        #adding precision recall data
        worksheet_pr.write(0,0, "Precision KFold")
        worksheet_pr.write(0,1, "Recall KFold")
        worksheet_pr_val.write(0,0, "Precision Val")
        worksheet_pr_val.write(0,1, "Recall Val")
        for row_num in range(len(precision_20)):
            temp = worksheet_pr.write(row_num+1,0,precision_20[row_num])
            temp = worksheet_pr.write(row_num+1,1,recall_20[row_num])
        for row_num in range(len(precision_val_20)):
            temp = worksheet_pr_val.write(row_num+1,0,precision_val_20[row_num])
            temp = worksheet_pr_val.write(row_num+1,1,recall_val_20[row_num])

        # adding ROC tpr and fpr axes
        worksheet_roc.write(0,0, "TPR KFold")
        worksheet_roc.write(0,1, "FPR KFold")
        worksheet_roc_val.write(0,0, "TPR Val")
        worksheet_roc_val.write(0,1, "FPR Val")
        for row_num in range(len(tpr_20)):
            temp = worksheet_roc.write(row_num+1,0,tpr_20[row_num])
            temp = worksheet_roc.write(row_num+1,1,fpr_20[row_num])
        for row_num in range(len(tpr_val_20)):
            temp = worksheet_roc_val.write(row_num+1,0,tpr_val_20[row_num])
            temp = worksheet_roc_val.write(row_num+1,1,fpr_val_20[row_num])

    workbook.close()