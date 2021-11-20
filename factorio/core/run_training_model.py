import configparser
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributions.poisson import Poisson
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Subset
from factorio.gpmodels.gppoissonpl import RateGPpl, fit
from factorio.utils import data_loader
from factorio.utils.helpers import percentiles_from_samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    # Move to config at some point
    dtype = torch.float
    num_inducing = 64
    num_iter = 200
    num_particles = 32
    loader_batch_size = 15000
    learn_inducing_locations = True
    slow_mode = False  # enables checkpointing and logging
    learning_rate = 0.0001

    time_now = datetime.datetime.utcnow()
    parser = argparse.ArgumentParser()

    path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                      help='Set path to your config.ini file.')
    path_parser = parser.add_argument('-o', '--output', type=Path, default='mnt/model_state.pth',
                                      help='Set path to load saved model.')

    args = parser.parse_args()
    if not args.config.exists():
        raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                                  f"to config.ini file, please check it!")
    output_path = args.output

    hack_config = data_loader.HackConfig.from_config(args.config)
    data = data_loader.load_data(hack_config.z_case)
    dfactory = data_loader.DataFactory(data,
                                       hack_config.data_frequency,
                                       teams=hack_config.teams,
                                       hospital=hack_config.hospital,
                                       data_folder=hack_config.data_folder,
                                       dtype=dtype)

    X_mins, X_maxs = dfactory.get_min_max()

    my_inducing_pts = torch.stack([
        torch.linspace(minimum, maximum, num_inducing, dtype=dtype)
        for minimum, maximum in zip(X_mins, X_maxs)
    ], dim=-1)

    dlen = len(dfactory.dset)
    loader = DataLoader(
        Subset(dfactory.dset, torch.arange(dlen-1000, dlen)-1),
        batch_size=loader_batch_size,
        shuffle=True
    )
    model = RateGPpl(inducing_points=my_inducing_pts,
                     learn_inducing_locations=learn_inducing_locations,
                     num_particles=num_particles,
                     lr=learning_rate)

    fit(model,
        train_dataloader=loader,
        max_epochs=num_iter,
        patience=10,
        verbose=False,
        enable_checkpointing=slow_mode,
        enable_logger=True)
    
    model.save_model(output_path)

    test_x = dfactory.dset[-200:][0]
    Y = dfactory.dset[-200:][1]
    x_plt = torch.arange(Y.size(0)).detach().cpu()
    model.eval()
    with torch.no_grad():
        output = model(test_x)

    # Similarly get the 5th and 95th percentiles
    samples = output(torch.Size([1000])).exp()
    lower, fn_mean, upper = percentiles_from_samples(samples)

    y_sim_lower, y_sim_mean, y_sim_upper = percentiles_from_samples(
        Poisson(samples).sample())

    # visualize the result
    fig, (ax_func, ax_samp) = plt.subplots(1, 2, figsize=(12, 3))
    line = ax_func.plot(
        x_plt, fn_mean.detach().cpu(), label='GP prediction')
    ax_func.fill_between(
        x_plt, lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(), color=line[0].get_color(), alpha=0.5
    )
    ax_func.legend()

    ax_samp.scatter(x_plt, Y, alpha=0.5,
                    label='True train data', color='orange')
    y_sim_plt = ax_samp.plot(x_plt, y_sim_mean.cpu(
    ).detach(), alpha=0.5, label='Sample mean from the model')
    ax_samp.fill_between(
        x_plt, y_sim_lower.detach().cpu(),
        y_sim_upper.detach().cpu(), color=y_sim_plt[0].get_color(), alpha=0.5
    )
    ax_samp.legend()
    plt.show()

    print(f'Done')