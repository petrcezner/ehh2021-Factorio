import math
from typing import Tuple
from pyro.distributions import constraints
import pyro.distributions as dist
import torch
from torch.distributions.poisson import Poisson
from torch.utils.data import TensorDataset, DataLoader
from factorio.utils.helpers import percentiles_from_samples
import pyro
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from tqdm import trange


class RateGP(gpytorch.models.ApproximateGP):
    def __init__(self,
                 inducing_points: torch.Tensor,
                 name_prefix="rate_exact_gp",
                 learn_inducing_locations=False,
                 lb_periodicity=0,
                 kernel=None):
        self.name_prefix = name_prefix
        # Define all the variational stuff
        num_inducing = inducing_points.size(0)
        ard_num_dims = inducing_points.size(1)
        variational_dist = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points,
            variational_dist,
            learn_inducing_locations=learn_inducing_locations
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean(ard_num_dims=ard_num_dims)
        if kernel is None:
            # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims))
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        self.covar_module = kernel  # \
        # + gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(
        #     period_length_constraint= gpytorch.constraints.GreaterThan(lb_periodicity)))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp", self)

        # register the variational parameters with Pyro.
        # ss_offset = pyro.param("ss_offset_q",
        #                        torch.tensor(0.224),
        #                        constraint=constraints.positive)
        # noise_scale = pyro.param("process_noise_scale_q",
        #                          torch.tensor(1.01),
        #                          constraint=constraints.positive)

        # Get p(f) - prior distribution of latent function
        rate_gp_lat = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", rate_gp_lat)

            # Use the link function to convert GP samples into observations dists parameters
            rate_positive = function_samples.exp()

            # transforms = [dist.transforms.ExpTransform(), dist.transforms.AffineTransform(loc=ss_offset, scale=1.0)]
            # transformed_dist = dist.TransformedDistribution(
            #     dist.Normal(transformed_samples, noise_scale),
            #     transforms
            #     )

            # Sample from observed distribution
            return pyro.sample(
                self.name_prefix + ".y",
                Poisson(rate_positive),
                obs=y
            )

    def log_prob(self, x, y):
        output = self(x)
        mean = output.mean
        return Poisson(mean).log_prob(y)

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(f'Run {__file__}')

    # Here we specify a 'true' latent function lambda
    lat_fn = lambda x: torch.sin(2 * math.pi * x) + torch.sin(3.3 * math.pi * x)
    obs_fn = lambda x: Poisson(x.exp()).sample()

    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 1000
    print(f'NSamp = {NSamp}')
    time_range = (0, 2.5)

    # X = torch.linspace(time_range[0], time_range[1], NSamp)
    X = torch.stack([
        torch.linspace(time_range[0], time_range[1], NSamp),
        torch.randn(NSamp)
    ], dim=-1).float()
    fx = lat_fn(X[:, 0])
    Y = obs_fn(fx).float()

    fig, (ax_lat, ax_sample) = plt.subplots(1, 2, figsize=(10, 3))
    ax_lat.plot(X[:, 0], fx.exp())
    ax_lat.set_xlabel('x')
    ax_lat.set_ylabel('$f(x)$')
    ax_lat.set_title('Latent function')
    ax_sample.scatter(X[:, 0], Y)
    ax_sample.set_xlabel('x')
    ax_sample.set_ylabel('y')
    ax_sample.set_title('Observations with Noise')
    plt.show()

    my_inducing_pts = torch.stack([
        torch.linspace(time_range[0], time_range[1], 32),
        torch.randn(32)
    ], dim=-1)
    model = RateGP(inducing_points=my_inducing_pts)
    model.fit(X, Y, num_iter=1000, num_particles=64)

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
