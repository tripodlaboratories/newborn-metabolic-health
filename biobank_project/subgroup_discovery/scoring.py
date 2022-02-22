import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class SubgroupScorer():
    def __init__(
        self,
        sg_num,
        sg_description,
        sg_mask: 'np.ndarray[np.bool]',
        true_vals: np.ndarray,
        preds: np.ndarray,
        multiple_iter_preds: pd.DataFrame=None):
        '''
        args:
            multiple_iter_preds:
                Dataframe of predictions, one column per iteration (e.g., per repeated K-Fold iteration)
        '''
        self.sg_num = sg_num
        self.sg_description = sg_description
        self.sg_mask = sg_mask
        self.true_vals = true_vals[sg_mask]
        self.preds = preds[sg_mask]
        if multiple_iter_preds is not None:
            self.multiple_iter_preds = multiple_iter_preds
            self.n_iters = len(multiple_iter_preds.columns)

        self.logger = logging.getLogger('SubgroupScorer')

    def _no_subgroup_exceptions(self):
        # Need to handle the case where a subgroup is NOT FOUND in a (likely validation)
        # dataset OR there are no true positives
        if self.sg_mask.sum() == 0:
            self.logger.warning(
                f'No individuals found in subgroup {self.sg_num}: {self.sg_description}')
            return False
        elif len(self.true_vals.unique()) == 1:
            self.logger.warning(
                f'Only one case type present in subgroup {self.sg_num}: {self.sg_description}')
            return False
        else:
            return True

    def _score_overall_auprc(self, specified_preds=None, return_tuple=False):
        if self._no_subgroup_exceptions():
            preds = specified_preds if specified_preds is not None else self.preds
            precision, recall, thresholds = precision_recall_curve(
                    self.true_vals, preds)
            if return_tuple is False:
                return auc(recall, precision)
            else:
                return (precision, recall, thresholds)
        else:
            if return_tuple is False:
                return np.nan
            else:
                # null tuple in place of (recall, precision, thresholds)
                # in list form since usually arrays of recall, precision, and
                # thresholds are returned
                return ([np.nan], [np.nan], [np.nan])

    def _score_multiple_iters_auprc(self):
        auprc_over_iters = []
        for i in range(self.n_iters):
            iter_auprc = self._score_overall_auprc(
                specified_preds=self.multiple_iter_preds.iloc[:, i][self.sg_mask])
            auprc_over_iters.append(iter_auprc)
        AUPRC_mean = np.mean(auprc_over_iters)
        if len(auprc_over_iters) == 1:
            AUPRC_sd = 0
        else:
            AUPRC_sd = np.std(auprc_over_iters, ddof=1)

        return AUPRC_mean, AUPRC_sd

    def score_auprc(self, score_over_iters=False):
        if score_over_iters is False:
            return self._score_overall_auprc()
        else:
            return self._score_multiple_iters_auprc()

    def _score_overall_auroc(self, specified_preds=None, return_tuple=False):
        if self._no_subgroup_exceptions():
            preds = specified_preds if specified_preds is not None else self.preds
            if return_tuple is False:
                return roc_auc_score(self.true_vals, preds)
            else:
                return roc_curve(self.true_vals, preds)
        else:
            if return_tuple is False:
                return np.nan
            else:
                # A null tuple instead of (fpr, tpr, thresholds)
                # in list form to match usually array return types
                return ([np.nan], [np.nan], [np.nan])

    def _score_multiple_iters_auroc(self):
        auroc_over_iters = []
        for i in range(self.n_iters):
            iter_auroc = self._score_overall_auroc(
                specified_preds=self.multiple_iter_preds.iloc[:, i][self.sg_mask])
            auroc_over_iters.append(iter_auroc)
        AUROC_mean = np.mean(auroc_over_iters)
        if len(auroc_over_iters) == 1:
            AUROC_sd = 0
        else:
            AUROC_sd = np.std(auroc_over_iters, ddof=1)

        return AUROC_mean, AUROC_sd

    def score_auroc(self, score_over_iters=False):
        if score_over_iters is False:
            return self._score_overall_auroc()
        else:
            return self._score_multiple_iters_auroc()
