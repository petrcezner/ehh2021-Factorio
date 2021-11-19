from collections import Iterable

from factorio.gpmodels.gppoissonpl import RateGPpl
from factorio.utils import helpers


class GpProblem:
    def __init__(self, dset, ard_num_dims,
                 cv_ratios: Iterable = None,
                 max_train_iter: int = 1000,
                 train_samples: int = 300):
        if cv_ratios is None:
            cv_ratios = [0.5]
        self.cv_ratios = cv_ratios
        self.dset = dset
        self.ard_num_dims = ard_num_dims
        self.valid_size = 2
        self.train_samples = train_samples

    def run_experiment(self, params_dict):
        kernel_names = [val for key, val in params_dict.items()
                        if 'ker' in key]
        kernel = helpers.generate_additive_kernel(kernel_names,
                                                  ard_num_dims=self.ard_num_dims)

        cv_datasets = helpers.timeseries_split(self.dset,
                                               self.cv_ratios,
                                               valid_size=self.valid_size,
                                               train_samples=self.train_samples)

        train_dataset, _ = cv_datasets[0]

        rate_gp = RateGPpl()
