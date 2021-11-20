import re
from typing import Iterable
from threading import Timer
from functools import reduce

import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, RQKernel, PeriodicKernel, SpectralMixtureKernel, \
    PolynomialKernel
import torch
from torch.utils.data import Dataset, Subset


class RepeatedTimer():
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def generate_additive_kernel(kernel_names: Iterable,
                             ard_num_dims: int = 1):
    kers_list = [get_kernel_by_name(ker_name, ard_num_dims)
                 for ker_name in kernel_names if 'None' not in ker_name]
    assert len(kers_list) > 0, f'{len(kers_list)} kernels provided, needs at least one.'
    return reduce(lambda a, b: a + b, kers_list)


def get_kernel_by_name(ker_name: str, ard_num_dims: int = 1):
    ker = None
    if 'matern' in ker_name:
        ker = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_num_dims))
    if 'matern15' in ker_name:
        ker = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_num_dims))
    if 'matern05' in ker_name:
        ker = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=ard_num_dims))
    if 'rbf' in ker_name:
        ker = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))
    if 'rq' in ker_name:
        ker = ScaleKernel(RQKernel(ard_num_dims=ard_num_dims))
    if 'periodic' in ker_name:
        ker = ScaleKernel(PeriodicKernel(
            ard_num_dims=ard_num_dims,
            period_length_constraint=gpytorch.constraints.Interval(
                lower_bound=.98,
                upper_bound=1.05
            )
        ))
    if 'spectral' in ker_name or 'poly' in ker_name:
        re_num = re.compile(r'.*_(\d+)')
        num_res = re_num.findall(ker_name)
        num = int(num_res[0])

        if 'spectral' in ker_name:
            ker = SpectralMixtureKernel(num_mixtures=num, ard_num_dims=ard_num_dims)
        if 'poly' in ker_name:
            ker = ScaleKernel(PolynomialKernel(power=num, ard_num_dims=ard_num_dims))
    return ker


def timeseries_split(dataset: Dataset,
                     ratios: Iterable = None,
                     valid_size: int = 1,
                     train_samples=0):
    if ratios is None:
        ratios = [0.5]
    length = len(dataset)
    parts = set([int(r * length) for r in ratios])
    parts = [part for part in parts if part + valid_size <= length and part - train_samples >= 0]
    parts.sort()
    tr_val_tuples = [
        (Subset(dataset, range(part - train_samples, part)),
         Subset(dataset, range(part, part + valid_size)))
        for part in parts
    ]
    return tr_val_tuples


def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
    """A quick helper function for getting smoothed percentile values from samples

    Args:
        samples (torch.Tensor): samples (y values) from distribution
        percentiles (list, optional): Percentiles to approximate. Defaults to [0.05, 0.5, 0.95].

    Returns:
        list: list of MC estimated percentiles queried by `percentiles`
    """    
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

    # Smooth the samples
    kernel = torch.full((1, 1, 5), fill_value=0.2)
    percentiles_samples = [
        torch.nn.functional.conv1d(percentile_sample.view(1, 1, -1), kernel, padding=2).view(-1)
        for percentile_sample in percentile_samples
    ]

    return percentiles_samples