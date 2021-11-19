import configparser
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from factorio.utils import data_loader


if __name__ == '__main__':
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
    for i in DataLoader(dfactory.dset):
        print(i)
