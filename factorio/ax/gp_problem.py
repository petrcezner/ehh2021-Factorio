from factorio.utils import helpers


class GpProblem:
    def __init__(self, dset, ard_num_dims):
        self.dset = dset
        self.ard_num_dims = ard_num_dims

    def run_experiment(self, params_dict):
        kernel_names = [val for key, val in params_dict.items()
                        if 'ker' in key]
        kernel = helpers.generate_additive_kernel(kernel_names,
                                                  ard_num_dims=self.ard_num_dims)
        #
        # cv_datasets = helpers.timeseries_split(self.dataset,
        #                                        ratios,
        #                                        valid_size=valid_size,
        #                                        train_days=train_days)
