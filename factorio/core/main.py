import configparser
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from factorio.gpmodels.gppoissonpl import RateGPpl, fit
from factorio.utils import data_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    time_now = datetime.datetime.utcnow()
    parser = argparse.ArgumentParser()

    path_parser = parser.add_argument('-c', '--config', type=Path, default='config.ini',
                                      help='Set path to your config.ini file.')

    args = parser.parse_args()
    if not args.config.exists():
        raise argparse.ArgumentError(path_parser, f"Config file doesn't exist! Invalid path: {args.config} "
                                                  f"to config.ini file, please check it!")

    hack_config = data_loader.HackConfig.from_config(args.config)
    data = data_loader.load_data(hack_config.z_case)
    dfactory = data_loader.DataFactory(data, hack_config.data_frequency)
    # for i in DataLoader(dfactory.dset):
    #     print(i)

    # subset = dfactory.dset[-1000:]
    # plt.plot(subset[0], subset[1])
    # plt.show()

    # Move to config at some point
    num_inducing = 128
    num_iter = 1000
    num_particles = 32
    loader_batch_size = 256
    slow_mode = False  # enables checkpointing and logging

    # time_range = dfactory.dset[0][0][0], dfactory.dset[0][0][-1]
    X_mins = [0., 0]
    X_maxs = [1000., 1.]

    my_inducing_pts = torch.stack([
        torch.linspace(minimum, maximum, num_inducing)
        for minimum, maximum in zip(X_mins, X_maxs)
    ], dim=-1)
    model = RateGPpl(inducing_points=my_inducing_pts,
                     num_particles=num_particles)
    loader = DataLoader(
        dfactory.dset,
        batch_size=loader_batch_size,
        shuffle=True
    )

    fit(model,
        train_dataloader=loader,
        max_epochs=num_iter,
        patience=10,
        verbose=False,
        enable_checkpointing=slow_mode,
        enable_logger=True)