"""Script for running deep multi-task experiments."""
import argparse
import logging
import os
from pathlib import PurePath

import pandas as pd
from torch import optim, Tensor
from torch.nn import BCEWithLogitsLoss

from biobank_project.deep_mtl.training import handlers, kfold, utils
from biobank_project.deep_mtl.models import base, ensemble
from biobank_project.deep_mtl.models import bottleneck
from biobank_project.deep_mtl.sampling import MajorityDownsampler


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Script for running deep multi-task feature removal experiments.',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input', type=str, help='input file with features and outcomes',
        required=True, metavar='INPUT_FILE', dest='input_file')
    parser.add_argument(
        '-o', '--output', type=str, help='output directory for results',
        required=True, metavar='OUTPUT_DIR', dest='output_dir')
    parser.add_argument(
        '-n', '--n_iter', type=int, help='number of cross validation iterations to perform',
        default=5)
    parser.add_argument(
        '--cases_only', action='store_true',
        help='perform model training on cases only, ignoring controls.')
    parser.add_argument(
        '--total_aupr_ranks', action='store_true',
        help='Use total aupr rankings instead of individual outcome AUPR ranks')
    parser.add_argument(
        '--random_feature_order', action='store_true',
        help='Instead of using feature ranks, randomize the feature order')
    return parser


def read_lines(file) -> list:
    with open(file) as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


def write_lines(list_to_write: list, filename: str) -> None:
    with open(filename, 'w') as f:
        f.writelines(item + '\n' for item in list_to_write)


def write_results(results: dict, model_output_dir: PurePath):
    for result_name, results_df in results.items():
        if result_name == 'preds':
            filename = model_output_dir.joinpath(result_name + '.csv.gz')
            results_df.to_csv(filename, compression='gzip')
        else:
            filename = model_output_dir.joinpath(result_name + '.csv')
            results_df.to_csv(filename)


def main(args):
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO'),
        format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
    logger = logging.getLogger('deep_mtl_experiment')
    input_file = args.input_file
    output_dir = PurePath(args.output_dir)
    n_iter = args.n_iter
    cases_only = args.cases_only
    total_aupr_ranks = args.total_aupr_ranks
    random_feature_order = args.random_feature_order

    # Read in data
    input_data = pd.read_csv(input_file, low_memory=False)
    input_data.set_index('row_id', inplace=True)
    metadata = pd.read_csv('./data/processed/metadata.csv', low_memory=False)
    metadata.set_index('row_id', inplace=True)
    outcomes = read_lines('./config/neonatal_covariates.txt')

    # Read in AUPR rankings for each neonatal outcome
    score_dir = PurePath('./results/neonatal/metabolite_scores/')
    auroc_rankings = pd.read_csv(
        score_dir.joinpath('neonatal_metabolite_auroc_no_overlap.csv'))
    auroc_rankings.loc[:, outcomes] = abs(0.5 - auroc_rankings.loc[:, outcomes])
    features_by_auroc = {
        k: auroc_rankings.sort_values(k, ascending=False)[['metabolite', k]]
        for k in outcomes}
    aupr_rankings = pd.read_csv(
        score_dir.joinpath('neonatal_metabolite_aupr_no_overlap.csv'))
    # Add a total AUPR ranking
    aupr_rankings['total_aupr'] = aupr_rankings[outcomes].sum(axis=1)
    features_by_aupr = {
        k: aupr_rankings.sort_values(k, ascending=False)[['metabolite', k]]
        for k in outcomes + ['total_aupr']}

    # Randomize order for given option
    if random_feature_order is True:
        logger.info('Using randomized feature order for removal experiment.')
        random_ranks = aupr_rankings.sample(frac=1, random_state=101)
        features_by_aupr = {k: random_ranks[['metabolite', k]]
                            for k in outcomes + ['total_aupr']}

    # A random classifier gives AUPR equal to proportion of pos over total
    pos_prop = pd.read_csv(
        score_dir.joinpath('outcome_pos_frac_no_overlap.csv'))

    # Subset data based on gestational ages used
    included_ga_range = read_lines('./config/gestational_age_ranges.txt')
    included_metadata = metadata[metadata['gacat'].isin(included_ga_range)]
    input_data = input_data.loc[included_metadata.index, :]
    assert sorted(input_data.index) == sorted(included_metadata.index)

    # Drop sparse columns
    input_data.dropna(thresh=len(input_data) / 2, axis=1, inplace=True)
    input_data.dropna(inplace=True)

    # Handling for cases only
    if cases_only is True:
        input_data['total_conditions'] = input_data[outcomes].sum(axis=1)
        data_subset = input_data[input_data['total_conditions'] >= 1].drop(
            columns=['total_conditions'])
        if data_subset.shape[0] >= input_data.shape[0]:
            logger.warn('--cases_only flag did not reduce rows of data.')
        input_data = data_subset
        assert 'total_conditions' not in input_data

    # Split data into X and Y
    data_X = input_data.drop(outcomes, axis=1)
    data_Y = input_data[outcomes]

    # Train with different models
    utils.seed_torch(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    n_hidden = 100

    # Set up model training
    batch_size = 3000
    shuffle_batch = True
    n_epochs = 50
    pos_weight = Tensor(data_Y.apply(utils.get_pos_weight))
    early_stopping_patience = 5
    early_stopping_handler = handlers.EarlyStopping(
        patience=early_stopping_patience)
    resampler = None

    os.makedirs(output_dir, exist_ok=True)

    if total_aupr_ranks is True:
        ranks = ['total_aupr']
    else:
        ranks = outcomes

    for rank_order in ranks:
        ranked_feature_list = features_by_aupr[rank_order]['metabolite'].copy().tolist()
        ranked_features = [f for f in ranked_feature_list if f in data_X.columns]

        while len(ranked_features) > 0:
            n_features = len(ranked_features)
            all_models = {
            'ensemble': ensemble.EnsembleNetwork(
                n_features=n_features, n_hidden=n_hidden, n_tasks=n_tasks),
            'ensemble_bottleneck_10': bottleneck.EnsembleNetwork(
                n_features=n_features, n_tasks=n_tasks,
                n_hidden=n_hidden, n_bottleneck=10)
            }

            for model_name, model in all_models.items():
                training_handler = handlers.ModelTraining(
                    model=model, batch_size=batch_size,
                    shuffle_batch=shuffle_batch, optimizer_class=optim.Adam)
                logger.info(' '.join((
                    'Training', model_name, 'with', str(n_features),
                    'features.')))
                X_subset = data_X[ranked_features]
                train_args = {
                    'n_epochs': n_epochs,
                    'criterion': BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight),
                    'colnames': data_Y.columns,
                    'early_stopping_handler': early_stopping_handler,
                    'output_training_preds': True
                    }
                kfold_handler = kfold.RepeatedKFold(
                    n_iter=n_iter, n_folds=5, data_X=X_subset, data_Y=data_Y,
                    training_handler=training_handler)
                model_results = kfold_handler.repeated_kfold(
                    training_args=train_args, resampler=resampler)
                model_results['scores']['n_features'] = n_features

                logger.info('Finished model training for: ' + model_name)

                # Summarize and store all the results along the way.
                if total_aupr_ranks is True:
                    experiment_output = '_'.join(
                        (model_name, str(n_features), 'feat', 'total_aupr_ranks'))
                else:
                    experiment_output = '_'.join(
                        (model_name, str(n_features), 'feat', rank_order))

                model_output_dir = output_dir.joinpath(experiment_output + '/')
                os.makedirs(model_output_dir, exist_ok=True)
                write_results(model_results, model_output_dir)
                write_lines(ranked_features, model_output_dir.joinpath('features.txt'))
                logger.info('Model results written to: ' + str(model_output_dir) + '/')

            # Remove least informative feature and continue training iterations
            ranked_features.pop()


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
