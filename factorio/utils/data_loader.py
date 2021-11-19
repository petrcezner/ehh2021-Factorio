import configparser
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


class DataLoader:
    def __init__(self, data, data_frequency):
        self.data = self.create_timestamp(data, data_frequency)

    def create_timestamp(self, data, data_frequency):
        data_ikem = data[data['destination__hospitalId'] == 'hospital:IKEM']
        cols = ['createdTs', 'closedTs', 'ambulanceLocation__first__dispatchingEtaTs', 'dispatchingTs']
        for col in cols:
            data_ikem[col] = pd.to_datetime(data_ikem[col])
        time_data = data_ikem[cols + ['ambulanceLocation__first__dispatchingEtaTs__minutes']]
        time_data.reset_index(drop=True, inplace=True)
        tmp_array = np.full((time_data.shape[0], 1), 0)
        time_data['cases'] = tmp_array
        for i, row in time_data.iterrows():
            if row['ambulanceLocation__first__dispatchingEtaTs'] is pd.NaT:
                time_data.loc[i, 'ambulanceLocation__first__dispatchingEtaTs'] = time_data.loc[i, 'dispatchingTs']
        time_data.set_index('ambulanceLocation__first__dispatchingEtaTs', inplace=True, drop=True)
        hour_rate = time_data.resample(f'{data_frequency}min').count()
        hour_rate['timedelta'] = np.linspace(0, 1000, hour_rate.shape[0])
        hour_rate['cases'].plot()
        plt.show()
        return TensorDataset(torch.as_tensor(hour_rate['timedelta'].values),
                             torch.as_tensor(hour_rate['cases'].values))


def load_data(data_path):
    data = pd.read_json(data_path, lines=True)
    return data


class HackConfig:
    def __init__(self, z_case, data_frequency):
        self.z_case = z_case
        self.data_frequency = data_frequency

    @classmethod
    def from_config(cls, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        z_case = Path(config['IKEM'].get('data_path'))
        data_frequency = config['IKEM'].getint('data_frequency', fallback=60)

        return cls(z_case=z_case,
                   data_frequency=data_frequency)


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

    hack_config = HackConfig.from_config(args.config)
    data = load_data(hack_config.z_case)
    data_loader = DataLoader(data, hack_config.data_frequency)
