import math
from typing import Tuple, Iterable
import gpytorch
import pyro
import torch
from torch.distributions.poisson import Poisson
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.functional import Tensor
import torch.optim as optim
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from factorio.gpmodels.gppoisson import RateGP
from pathlib import Path


class RateGPpl(LightningModule):
    def __init__(self,
                 inducing_points: torch.Tensor,
                 name_prefix="mixture_gp",
                 learn_inducing_locations=False,
                 lr=0.01,
                 num_particles=64,
                 kernel=None):
        super().__init__()
        self.automatic_optimization = False

        self.gp = RateGP(inducing_points=inducing_points,
                         name_prefix=name_prefix,
                         learn_inducing_locations=learn_inducing_locations,
                         kernel=kernel)
        self.lr = lr
        self.num_particles = num_particles
        self.save_hyperparameters()

    def forward(self, x):
        output = self.gp(x)
        return output

    # def predict(self, X):
    #     return self.gp.predict(X)

    def configure_optimizers(self):
        optimizer = pyro.optim.Adam({"lr": self.lr})
        elbo = pyro.infer.TraceGraph_ELBO(
            num_particles=self.num_particles, vectorize_particles=True, retain_graph=True)
        self.svi = pyro.infer.SVI(
            self.gp.model, self.gp.guide, optimizer, elbo)
        self.train()
        return None

    def training_step(self, batch, batch_idx, *args, **kwargs):
        tr_x, tr_y = batch
        self.zero_grad()
        loss = torch.as_tensor(self.svi.step(tr_x, tr_y))
        # Output from model
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def eval_performance(self, val_dset: Iterable[Tuple[Dataset, Dataset]]):
        results = []

        for tr, val in val_dset:
            x, y = val[:]
            self.eval()
            with torch.no_grad():
                output = self(x)
            lpd = self.log_prob(x, y)
            y_hat = Poisson(output).mean
            y_err = y - y_hat
            mae = (y_err.abs() / y).mean()
            maxabserr = (y_err.abs() / y).max()
            rmse = y_err.norm() / y.norm()
            err_last_sample = y_err[-1].abs() / y[-1]

            res = {
                'lpd': lpd,
                'rmse': rmse,
                'maxabserr': maxabserr,
                'mae': mae,
                'err_last_day': err_last_sample,
            }
            results.append(res)

        res_dict = {}
        keys = res.keys()
        for key in keys:
            res_dict[key] = torch.stack([
                res[key]
                for res in results
            ])
        return res_dict

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    @classmethod
    def load_model(cls, load_path, num_particles=32):
        loaded_state_dict = torch.load(load_path)
        loaded_inducing_points = loaded_state_dict['gp.variational_strategy.inducing_points']
        model = cls(inducing_points=loaded_inducing_points,
                        num_particles=num_particles)
        model.load_state_dict(loaded_state_dict)
        return model


def fit(module,
        train_dataloader,
        max_epochs=1000,
        patience=10,
        min_delta=1e-4,
        verbose=True,
        enable_logger=True,
        enable_checkpointing=True,
        use_gpu=False,
        gpus=-1):
    '''Runs training with earlystopping and constructs default trainer for you.'''
    callbacks = [
        EarlyStopping(
            monitor='train_loss',
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode='min',
            check_on_train_epoch_end=True
        )
    ]

    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            save_top_k=1,
            mode='min',
        )
        callbacks += [checkpoint_callback]
    if not use_gpu:
        gpus = 0
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        checkpoint_callback=enable_checkpointing,
        logger=enable_logger,
        auto_select_gpus=use_gpu,
        gpus=gpus,
    )
    trainer.fit(module, train_dataloader)
    # gp_state_dict = module.gp.state_dict()
    # torch.save(gp_state_dict, Path(trainer.log_dir) / Path('gp_state_dict.pth'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from factorio.utils.helpers import percentiles_from_samples

    print(f'Run {__file__}')


    # Here we specify a 'true' latent function lambda
    def lat_fn(x): return torch.sin(2 * math.pi * x) + \
                          torch.sin(3.3 * math.pi * x)


    def obs_fn(x): return Poisson(x.exp()).sample()


    num_inducing = 164
    num_iter = 1
    num_particles = 32
    slow_mode = False  # enables checkpointing and logging
    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 1000
    print(f'NSamp = {NSamp}')
    time_range = (0, 2.5)

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
        torch.linspace(time_range[0], time_range[1], num_inducing),
        torch.randn(num_inducing)
    ], dim=-1)
    model = RateGPpl(inducing_points=my_inducing_pts,
                     num_particles=num_particles)

    loader = DataLoader(
        TensorDataset(
            X,
            Y
        ),
        batch_size=256,
        shuffle=True
    )
    fit(model,
        train_dataloader=loader,
        max_epochs=num_iter,
        patience=10,
        verbose=False,
        enable_checkpointing=slow_mode,
        enable_logger=True)

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

    # Similarly get the 5th and 95th percentiles
    samples = output(torch.Size([1000]))
    lower, fn_mean, upper = percentiles_from_samples(samples)

    y_sim_lower, y_sim_mean, y_sim_upper = percentiles_from_samples(
        Poisson(samples.exp()).sample())

    # visualize the result
    fig, (ax_func, ax_samp) = plt.subplots(1, 2, figsize=(12, 3))
    line = ax_func.plot(
        test_x[:, 0], fn_mean.detach().cpu(), label='GP prediction')
    ax_func.fill_between(
        test_x[:, 0], lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(), color=line[0].get_color(), alpha=0.5
    )

    ax_func.legend()

    # sample from p(y|D,x) = \int p(y|f) p(f|D,x) df (doubly stochastic)
    ax_samp.scatter(X[:, 0], Y, alpha=0.5,
                    label='True train data', color='orange')
    # ax_samp.plot(test_x[:, 0], y_sim.cpu().detach(), alpha=0.5, label='Mean from the model')
    y_sim_plt = ax_samp.plot(test_x[:, 0], y_sim_mean.cpu().detach(), alpha=0.5, label='Sample from the model')
    ax_samp.fill_between(
        test_x[:, 0], y_sim_lower.detach().cpu(),
        y_sim_upper.detach().cpu(), color=y_sim_plt[0].get_color(), alpha=0.5
    )
    ax_samp.legend()
    plt.show()

    print(f'Done')
