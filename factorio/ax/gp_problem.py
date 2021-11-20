from collections import Iterable

import torch
from torch import norm
from torch.utils.data import DataLoader, Subset

from factorio.gpmodels.gppoissonpl import RateGPpl, fit
from factorio.utils import helpers


class GpProblem:
    def __init__(self, datafactory, ard_num_dims,
                 cv_ratios: Iterable = None,
                 max_train_iter: int = 1000,
                 slow_mode=False,
                 num_particles=32):
        if cv_ratios is None:
            cv_ratios = [0.5]
        self.cv_ratios = cv_ratios
        self.data_factory = datafactory
        self.dset = datafactory.dset
        self.ard_num_dims = ard_num_dims
        self.max_train_iter = max_train_iter
        self.slow_mode = slow_mode
        self.num_particles = num_particles
        self.dtype = torch.float
        self.loader_batch_size = 15000
        self.learning_rate = 0.0001
        self.num_inducing = 164

    def run_experiment(self, params_dict, use_gpu):
        kernel_names = [val for key, val in params_dict.items()
                        if 'ker' in key]
        kernel = helpers.generate_additive_kernel(kernel_names,
                                                  ard_num_dims=self.ard_num_dims)

        train_samples = params_dict['train_samples']
        valid_size = params_dict['valid_size']
        cv_datasets = helpers.timeseries_split(self.dset,
                                               self.cv_ratios,
                                               valid_size=valid_size,
                                               train_samples=train_samples)

        train_dataset, _ = cv_datasets[0]
        dlen = len(self.dset)
        loader = DataLoader(train_dataset, shuffle=True, batch_size=self.loader_batch_size)
        loader = DataLoader(
            Subset(self.dset, torch.arange(dlen - 1000, dlen) - 1),
            batch_size=self.loader_batch_size,
            shuffle=True
        )
        X_mins, X_maxs = self.data_factory.get_min_max()

        inducing_pts = torch.stack([
            torch.linspace(minimum, maximum, self.num_inducing, dtype=self.dtype)
            for minimum, maximum in zip(X_mins, X_maxs)
        ], dim=-1)

        model = RateGPpl(kernel=kernel,
                         num_particles=self.num_particles,
                         inducing_points=inducing_pts,
                         lr=self.learning_rate)

        fit(model,
            train_dataloader=loader,
            max_epochs=self.max_train_iter,
            verbose=False,
            patience=10,
            enable_checkpointing=self.slow_mode,
            enable_logger=True,
            use_gpu=use_gpu)

        res = model.eval_performance(cv_datasets[1])

        return {
            'lpd': torch.std_mean(res['lpd']),
            'rmse': torch.std_mean(res['rmse']),
            'mae': torch.std_mean(res['mae']),
            'maxabserr': torch.std_mean(res['maxabserr']),
            'err_last_sample': torch.std_mean(res['err_last_sample']),
            'lpd_min': (res['rmse'].min(), 0.),
            'rmse_max': (res['rmse'].max(), 0.),
            'mae_max': (res['mae'].max(), 0.),
            'maxabserr_max': (res['maxabserr'].max(), 0.),
            'err_last_sample_max': (res['err_last_sample'].abs().max(), 0.),
        }

    def evaluate(self, parameters, use_gpu=False):
        multitrial_res_list = [
            self.run_experiment(parameters,
                                use_gpu=use_gpu)
            for _ in range(parameters['num_trials'])
        ]

        keys = multitrial_res_list[0].keys()

        aggregated_res = {
            key: torch.tensor([res_dict[key]
                               for res_dict in multitrial_res_list])
            for key in keys
        }
        return {
            'lpd': (aggregated_res['lpd'][:, 0].mean().item(), norm(aggregated_res['lpd'][:, 1]).item()),
            'rmse': ((aggregated_res['rmse'][:, 0].mean()).item(), norm(aggregated_res['rmse'][:, 1]).item()),
            'mae': ((aggregated_res['mae'][:, 0].mean()).item(), norm(aggregated_res['mae'][:, 1]).item()),
            'maxabserr': (
                (aggregated_res['maxabserr'][:, 0].mean()).item(), norm(aggregated_res['maxabserr'][:, 1]).item()),
            'err_last_day': (
                (aggregated_res['err_last_day'][:, 0].mean()).item(),
                norm(aggregated_res['err_last_day'][:, 1]).item()),
            'lpd_min': (aggregated_res['lpd_min'][:, 0].min().item(), norm(aggregated_res['lpd_min'][:, 1]).item()),
            'rmse_max': (
                (aggregated_res['rmse_max'][:, 0].max()).item(), norm(aggregated_res['rmse_max'][:, 1]).item()),
            'mae_max': ((aggregated_res['mae_max'][:, 0].max()).item(), norm(aggregated_res['mae_max'][:, 1]).item()),
            'maxabserr_max': (
                (aggregated_res['maxabserr_max'][:, 0].max()).item(),
                norm(aggregated_res['maxabserr_max'][:, 1]).item()),
            'err_last_day_max': ((aggregated_res['err_last_day_max'][:, 0].max()).item(),
                                 norm(aggregated_res['err_last_day_max'][:, 1]).item()),
        }
