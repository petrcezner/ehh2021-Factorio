from pathlib import Path
from typing import Iterable, List, Tuple
from ax.service.ax_client import AxClient

from factorio.ax.gp_problem import GpProblem
from factorio.utils.hack_config import HackConfig


problem = GpProblem()

def kernel_client(trials, valid_size):
    ax_client_ = AxClient()

    spectral_ker_names = []
    for i in range(1, 4):
        spectral_ker_names.append(f'spectral_{i}')

    poly_ker_names = []
    for i in range(1, 4):
        poly_ker_names.append(f'poly_{i}')

    spectral_ker_names.append('None')
    poly_ker_names.append('None')

    ax_client_.create_experiment(
        name="optimal_spectral_kernel",
        parameters=[
            {
                "name": "valid_size",
                "type": "fixed",
                "value": valid_size,
            },
            {
                "name": "num_trials",
                "type": "fixed",
                "value": trials,
            },
            {
                "name": "train_days",
                "type": "range",
                "bounds": [385, 396],
                "value_type": "int",
                "log_scale": False,
            },
            {
                'name': "ker1",
                'value_type': 'str',
                'type': 'choice',
                'values': ['matern', 'rbf'],
            },
            {
                'name': "ker2",
                'value_type': 'str',
                'type': 'choice',
                'values': ['rbf', 'matern15', 'matern05', 'None'],
            },
            {
                'name': "ker3",
                'value_type': 'str',
                'type': 'choice',
                'values': spectral_ker_names,
            },
            {
                'name': "train_mean",
                'value_type': 'bool',
                'type': 'choice',
                'values': [False],
            },
        ],
        objective_name='maxabserr_max',
        tracking_metric_names=['lpd',
                               'rmse',
                               'maxabserr',
                               'mae',
                               'err_last_day',
                               'lpd_min',
                               'rmse_max',
                               'mae_max',
                               'maxabserr_max',
                               'err_last_day_max'],
        minimize=True,
    )
    return ax_client_


if __name__ == '__main__':
    import numpy as np
    import datetime
    from tqdm import trange
    import argparse

    time_now = datetime.datetime.utcnow()
    parser = argparse.ArgumentParser()

    path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                      help='Set path to your config.ini file.')

    args = parser.parse_args()
    if not args.config.exists():
        raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                                  f"to config.ini file, please check it!")

    experiment_root = Path(f'.out/experiments/{time_now.strftime("%Y-%m-%d")}')
    experiment_root.mkdir(parents=True, exist_ok=True)

    hack_config = HackConfig.from_config(args.config)

    file_name = Path(f'{hack_config.experiment_name}_{time_now.strftime("%Y%d%m%H%M%S")}.json')
    full_path = str(experiment_root / file_name)
    ax_client = kernel_client(trials=hack_config.inter_trials,
                              valid_size=hack_config.validation_size)

    file_name = Path(f"{args.experiment_name}_{time_now.strftime('%Y%d%m%H%M%S')}.json")
    full_path = str(experiment_root / file_name)
    for i in trange(hack_config.trials):
        parameters, trial_index = ax_client.get_next_trial()
        results = problem.evaluate(parameters, use_gpu=hack_config.use_gpu)
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)
        ax_client.save_to_json_file(full_path)

    best_parameters, values = ax_client.get_best_parameters()

    print(f"best parameters: {best_parameters}")
    df = ax_client.generation_strategy.trials_as_df
    df.to_csv(str(experiment_root / Path('output_dataframe.csv')))
