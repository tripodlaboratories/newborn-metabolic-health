'''
Created on 14.04.2021

@author: Tony Culos, Martin Becker

TODO: target variables and estimates should not be in the target as actual data
'''
from asyncio.log import logger
from dataclasses import dataclass
import numbers
from collections import namedtuple
from functools import total_ordering
import numpy as np
import pysubgroup as ps
import sklearn.metrics as metrics #TODO: import of entire package may be uneccesary, see if attribute list can be used
import scipy.stats


@total_ordering
class PredictionTarget:
    statistic_types = ('size_sg', 'size_dataset', 'pos_sg', 'pos_dataset', 'neg_sg', 'neg_dataset', "metric_sg", "metric_dataset")

    def __init__(self, target_variable, target_estimate, eval_func=None, eval_dict=None, training_mask=None):
        self.target_variable = target_variable
        self.target_estimate = target_estimate
        self.eval_dict = eval_dict
        if not eval_dict is None:
            PredictionTarget.statistic_types = ('size_sg', 'size_dataset', 'pos_sg', 'pos_dataset', 'neg_sg', 'neg_dataset', "metric_sg", "metric_dataset") + tuple([x +"_sg" for x in eval_dict.keys()]) + tuple([x +"_dataset" for x in eval_dict.keys()])
        else:
            PredictionTarget.statistic_types = ('size_sg', 'size_dataset', 'pos_sg', 'pos_dataset', 'neg_sg', 'neg_dataset', "metric_sg", "metric_dataset")
        if eval_func is None:
            self.evaluation_metric = default_evaluation_metric
        elif not hasattr(metrics, eval_func.__name__):
            raise ValueError("eval_func passed must be from sklearn.metrics")
        else:
            # TODO: move evaluation metric to qualit function
            self.evaluation_metric = eval_func
        self.training_mask = training_mask

    def __repr__(self):
        return "T: " + str(self.target_variable) + "\nT_hat: " +str(self.target_estimate)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [self.target_variable, self.target_estimate]

    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup)
        size_dataset = data.shape[0]
        metric_sg = self.evaluation_metric(self.target_variable[cover_arr], self.target_estimate[cover_arr])
        metric_dataset = self.evaluation_metric(self.target_variable, self.target_estimate)
        return (size_sg, size_dataset, metric_sg, metric_dataset)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        if cached_statistics is None or not isinstance(cached_statistics, dict):
            statistics = dict()
        elif all(k in cached_statistics for k in PredictionTarget.statistic_types):
            return cached_statistics
        else:
            statistics = cached_statistics

        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)

        statistics['size_sg'] = size_sg
        statistics['size_dataset'] = data.shape[0]

        statistics['pos_sg'] = self.target_variable[cover_arr].sum()
        statistics['pos_dataset'] = self.target_variable.sum()
        statistics['neg_sg'] = (1 - self.target_variable[cover_arr]).sum()
        statistics['neg_dataset'] = (1 - self.target_variable).sum()

        statistics['metric_sg'] = self.evaluation_metric(self.target_variable[cover_arr], self.target_estimate[cover_arr])
        statistics['metric_dataset'] = self.evaluation_metric(self.target_variable, self.target_estimate)

        if not self.eval_dict is None:
            for key in self.eval_dict.keys():
                statistics[key+"_sg"] = self.eval_dict[key](self.target_variable[cover_arr], self.target_estimate[cover_arr])
                statistics[key+"_dataset"] = self.eval_dict[key](self.target_variable, self.target_estimate)

        return statistics


@dataclass
class PredictionQFNumericParams:
    size_sg: float
    metric_sg: float
    estimate: float


class PredictionQFNumeric(ps.BoundedInterestingnessMeasure):
    @staticmethod
    def prediction_qf_numeric(a, size_sg, metric_sg, invert):
        if invert:
            if metric_sg != 0:
                return size_sg ** a * (1.0/metric_sg)
            else:
                return float("inf") #TODO: when metric_sg = 0 and inverted just return inf, this assumes low metric is bad
        return size_sg ** a * (metric_sg)

    def __init__(
            self,
            a,
            invert=False,
            training_focused_qf=False,
            max_training_ratio_difference=None,
            training_ratio_difference_exponent=None,
            n_samples=None):
        if not isinstance(a, numbers.Number):
            raise ValueError(f'a is not a number. Received a={a}')

        self.a = a
        self.max_training_ratio_difference = max_training_ratio_difference
        self.training_ratio_difference_exponent = training_ratio_difference_exponent
        self.training_focused_qf = training_focused_qf
        self.n_samples = n_samples

        self.size=None
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'metric_sg')
        self.dataset_statistics = None
        self.all_target_variable = None
        self.all_target_estimate = None
        self.all_target_metric = None
        self.has_constant_statistics = False
        self.estimator = PredictionQFNumeric.OptimisticEstimator(self)

    def calculate_constant_statistics(self, data, target):

        self.size = len(data)
        self.all_target_variable = target.target_variable
        self.all_target_estimate = target.target_estimate
        self.all_target_metric = target.evaluation_metric(self.all_target_variable, self.all_target_estimate)

        self.all_training_mask = target.training_mask
        if target.training_mask is not None:
            self.all_target_ratio = target.training_mask.sum() / self.size

        self.has_constant_statistics = True
        estimate = self.estimator.get_estimate(self.size, self.a)
        self.dataset_statistics = PredictionQFNumericParams(self.size, self.all_target_metric, estimate)


    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        #dataset = self.dataset_statistics #can be used to compare all data AUC to subgroup AUC
        return PredictionQFNumeric.prediction_qf_numeric(self.a, statistics.size_sg, statistics.metric_sg, self.invert)


    def calculate_statistics(self, subgroup, target, data, statistics=None):

        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_variable), data)

        if self.all_training_mask is not None and self.training_focused_qf:
            # only calculate qf on dsignated training samples
            msk = np.zeros(data.shape[0], dtype=bool)
            msk[cover_arr] = True
            sg_target_variable = self.all_target_variable[msk & self.all_training_mask]
            sg_target_estimate = self.all_target_estimate[msk & self.all_training_mask]
        else:
            sg_target_variable = self.all_target_variable[cover_arr]
            sg_target_estimate = self.all_target_estimate[cover_arr]

        if sg_size > 0 and sg_target_variable.size > 0 and np.std(sg_target_variable) != 0:

            estimate = self.estimator.get_estimate(sg_size, self.a)

            if self.n_samples:

                msk_validation = (~self.all_training_mask & msk)
                n_validation = msk_validation.sum()

                split_size = min(n_validation, (msk & self.all_training_mask).sum()) - 2 # TODO: hacky

                u_labels = np.unique(sg_target_variable)
                # stratified splitting will fail when
                #   * there are less than 2 validation samples (TODO: should not be hardcoded to 2 but number of overall classes)
                #   * there are less than two classes (TODO: COULD be forced to the number of overall classes)
                #   * there are not at least 2 samples for each class (for train and test split)
                if \
                        split_size < 2 \
                        or len(u_labels) < 2 \
                        or np.any([(l == sg_target_variable).sum() < 2 for l in u_labels]):  # TODO: hacky

                    metric_sg_samples = [0] * self.n_samples

                else:
                    import sklearn.model_selection
                    # split = sklearn.model_selection.ShuffleSplit(
                    split = sklearn.model_selection.StratifiedShuffleSplit(
                        n_splits=self.n_samples,
                        train_size=split_size)

                    metric_sg_samples = []
                    for idx_sample, _ in split.split(sg_target_estimate.reshape(-1,1), sg_target_variable):

                        sg_target_variable_sample = sg_target_variable[idx_sample]
                        sg_target_estimate_sample = sg_target_estimate[idx_sample]

                        try:
                            metric_sg = target.evaluation_metric(sg_target_variable_sample, sg_target_estimate_sample)
                            metric_sg_samples.append(metric_sg)
                        except:
                            metric_sg_samples.append(0)


                metric_sg = np.median(metric_sg_samples)
                metric_sg_std = scipy.stats.median_absolute_deviation(metric_sg_samples)
                metric_sg_std_min = np.min(metric_sg)
                metric_sg_std_max = np.max(metric_sg)

            else:
                metric_sg = target.evaluation_metric(sg_target_variable, sg_target_estimate)
                metric_sg_std = 0
                metric_sg_std_min = metric_sg
                metric_sg_std_max = metric_sg

            if np.isnan(metric_sg):
                print(f"Invalid metric. {metric_sg} ({metric_sg_samples})")
                estimate = float('-inf')
                metric_sg = 0# float('-inf')

            # training focused constraints
            if self.max_training_ratio_difference is not None or self.training_ratio_difference_exponent is not None:

                msk = np.zeros(data.shape[0], dtype=bool)
                msk[cover_arr] = True
                n_training = (msk & self.all_training_mask).sum()
                training_ratio = n_training / sg_size
                training_ratio_difference = np.abs(training_ratio - self.all_target_ratio)

                # kick out any subgroup with a training sample ratio difference more than X
                if self.max_training_ratio_difference is not None and training_ratio_difference > self.max_training_ratio_difference:
                    estimate = float('-inf')
                    metric_sg = 0

                # weight subgroup according to training ratio difference
                elif self.training_ratio_difference_exponent is not None:
                    metric_sg = metric_sg * scipy.stats.norm.pdf((1 - training_ratio_difference) **self.training_ratio_difference_exponent)

        else:
            estimate = float('-inf')
            metric_sg = 0# float('-inf')

        return PredictionQFNumericParams(sg_size, metric_sg, estimate)


    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate

    class OptimisticEstimator:
        def __init__(self, qf):
            self.qf = qf
            self.metric = None

        def get_data(self, data):
            return data

        def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
            self.metric = float('inf')# target.evaluation_metric

        def get_estimate(self, size_sg, a):
            max_possible = 1
            return size_sg ** a * (max_possible) #TODO: how to extract max from all sklearn metrics dynamically


#default eval function is average sub ranking loss, see Duivesteijn & Thaele
def default_evaluation_metric(y_true, y_pred):
    sorted_true = y_true[np.argsort(y_pred)]
    numerator_sum = 0
    for i in range(len(y_true)):
        if sorted_true[i] == 1: numerator_sum += (sorted_true[:i+1] == 0).sum()
    return numerator_sum/y_true.sum()
