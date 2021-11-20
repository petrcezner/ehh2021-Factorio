import math
from typing import List, Tuple
from pyro.distributions import constraints
import pyro.distributions as dist
import torch
from torch.distributions.poisson import Poisson
from torch.utils.data import TensorDataset, DataLoader
from factorio.gpmodels.gppoisson import RateGP
from factorio.utils.helpers import percentiles_from_samples
import pyro
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.variational import VariationalStrategy
from tqdm import trange


class LogNormGP(gpytorch.models.pyro.PyroGP):
    def __init__(self,
                 num_inducing: int,
                 X_mins: List,
                 X_maxs: List,
                 name_prefix="rate_exact_gp",
                 learn_inducing_locations=False,
                 kernel=None,
                 num_data=100):
        self.name_prefix = name_prefix
        num_tasks = 2  # mean and stdvar of exponentiated distrib
        # Define all the variational stuff
        inducing_points = torch.stack([
            torch.linspace(minimum, maximum, num_inducing)
            for minimum, maximum in zip(X_mins, X_maxs)
        ], dim=-1)#.repeat(num_tasks, 1, 1)
        ard_num_dims = inducing_points.size(-1)
        variational_dist = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points,
            variational_dist,
            learn_inducing_locations=learn_inducing_locations
        )

        # Standard initializtation
        likelihood = LogNormLikelihood()
        super().__init__(variational_strategy, likelihood, num_data=num_data)
        self.likelihood = likelihood

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean(
            ard_num_dims=ard_num_dims
        )
        if kernel is None:
            # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims))
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=ard_num_dims
                ),
                ard_num_dims=ard_num_dims
            )
        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def fit(self, tr_x, tr_y, num_iter=100, num_particles=256):
        optimizer = pyro.optim.Adam({"lr": 0.01})
        elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True)
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, elbo)

        loader = DataLoader(
            TensorDataset(
                tr_x,
                tr_y
            ),
            batch_size=256,
            shuffle=True
        )

        self.train()
        iterator = trange(num_iter)
        for i in iterator:
            for x, y in loader:
                self.zero_grad()
                loss = svi.step(x, y)
                iterator.set_postfix(
                    loss=loss,
                    # lengthscale=self.covar_module.base_kernel.lengthscale.item(),
                )


class LogNormLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self):
        super().__init__()

        # # These are parameters/buffers for the cluster assignment latent variables
        # self.register_buffer("prior_cluster_logits", torch.zeros(num_tasks, num_clusters))
        # self.register_parameter("variational_cluster_logits", torch.nn.Parameter(torch.randn(num_tasks, num_clusters)))

        # # The Gaussian observational noise
        self.register_parameter("raw_noise", torch.nn.Parameter(torch.tensor(-5.0)))

        # Other info
        # self.num_tasks = num_tasks
        # self.max_plate_nesting = 1

    # def pyro_guide(self, function_dist, target):
    #     return super().pyro_guide(function_dist, target)

    # def pyro_model(self, function_dist, target):
    #     return super().pyro_model(function_dist, target)

    def forward(self, function_samples):
        locs = function_samples.exp()
        # Now we return the observational distribution, based on the function_samples and cluster_assignment_samples
        res = pyro.distributions.Normal(
            loc=locs,
            scale=self.raw_noise.exp()
        )
        return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    # Here we specify a 'true' latent function lambda
    lat_fn = lambda x: torch.sin(2 * math.pi * x) + torch.sin(3.3 * math.pi * x)
    obs_fn = lambda x: Poisson(x.exp()).sample()

    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 1000
    num_inducing = 64
    num_iter = 200
    num_particles = 32
    loader_batch_size = 15000
    learn_inducing_locations = True
    slow_mode = False  # enables checkpointing and logging
    learning_rate = 0.0001
    print(f'NSamp = {NSamp}')
    time_range = (0, 2.5)

    # X = torch.linspace(time_range[0], time_range[1], NSamp)
    X = torch.stack([
        torch.linspace(time_range[0], time_range[1], NSamp),
        torch.randn(NSamp)
    ], dim=-1).float()
    fx = lat_fn(X[:, 0])
    Y = obs_fn(fx).float()

    # fig, (ax_lat, ax_sample) = plt.subplots(1, 2, figsize=(10, 3))
    # ax_lat.plot(X[:, 0], fx.exp())
    # ax_lat.set_xlabel('x')
    # ax_lat.set_ylabel('$f(x)$')
    # ax_lat.set_title('Latent function')
    # ax_sample.scatter(X[:, 0], Y)
    # ax_sample.set_xlabel('x')
    # ax_sample.set_ylabel('y')
    # ax_sample.set_title('Observations with Noise')
    # plt.show()

    model = LogNormGP(num_inducing=num_inducing,
                      X_mins=[time_range[0], -2],
                      X_maxs=[time_range[1], 2],
                      learn_inducing_locations=True)
    model.fit(X,
              Y,
              num_iter=num_iter,
              num_particles=num_particles)

    # define test set (optionally on GPU)
    denser = 2  # make test set 2 times denser then the training set
    # test_x = torch.linspace(time_range[0], 2*time_range[1], denser * NSamp).float()#.cuda()
    test_x = torch.stack([
        torch.linspace(time_range[0], 2 * time_range[1], denser * NSamp),
        torch.randn(denser * NSamp)
    ], dim=-1).float()  # .cuda()

    model.eval()
    with torch.no_grad():
        output = model(test_x)

    # Get E[exp(f)] via f_i ~ GP, 1/n \sum_{i=1}^{n} exp(f_i).
    # Similarly get the 5th and 95th percentiles
    samples = output(torch.Size([1000]))
    lower, fn_mean, upper = percentiles_from_samples(samples)

    y_sim_lower, y_sim_mean, y_sim_upper = percentiles_from_samples(Poisson(samples.exp()).sample())

    # y_sim = obs_fn(fn_mean)

    # visualize the result
    fig, (ax_func, ax_samp) = plt.subplots(1, 2, figsize=(12, 3))
    line = ax_func.plot(test_x[:, 0], fn_mean.detach().cpu(), label='GP prediction')
    ax_func.fill_between(
        test_x[:, 0], lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(), color=line[0].get_color(), alpha=0.5
    )

    # func.plot(test_x, lat_fn(test_x), label='True latent function')
    ax_func.legend()

    # sample from p(y|D,x) = \int p(y|f) p(f|D,x) df (doubly stochastic)
    ax_samp.scatter(X[:, 0], Y, alpha=0.5, label='True train data', color='orange')
    # ax_samp.plot(test_x[:, 0], y_sim.cpu().detach(), alpha=0.5, label='Mean from the model')
    y_sim_plt = ax_samp.plot(test_x[:, 0], y_sim_mean.cpu().detach(), alpha=0.5, label='Sample from the model')
    ax_samp.fill_between(
        test_x[:, 0], y_sim_lower.detach().cpu(),
        y_sim_upper.detach().cpu(), color=y_sim_plt[0].get_color(), alpha=0.5
    )
    ax_samp.legend()
    plt.show()

    print(f'Done')
