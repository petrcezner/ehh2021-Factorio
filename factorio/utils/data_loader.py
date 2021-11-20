import configparser
import datetime
from pathlib import Path
import warnings

from factorio.utils.hack_config import HackConfig
from factorio.web_scraping.football import Football

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

import os

from factorio.weather import ActualWeather, HistoricalWeather

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataFactory:
    def __init__(self, data, data_frequency, teams, dtype=torch.float):
        self.teams = teams
        self.scaler = MinMaxScaler()
        self.dset = self.create_timestamp(data, data_frequency, dtype=dtype)

    def create_timestamp(self, data, data_frequency, dtype=torch.float):
        data_ikem = data[data['destination__hospitalId'] == 'hospital:IKEM']
        cols = ['createdTs', 'closedTs', 'ambulanceLocation__first__dispatchingEtaTs', 'dispatchingTs']
        for col in cols:
            data_ikem.loc[:, col] = pd.to_datetime(data_ikem[col])
        time_data = data_ikem[cols + ['ambulanceLocation__first__dispatchingEtaTs__minutes']]
        time_data.reset_index(drop=True, inplace=True)
        tmp_array = np.full((time_data.shape[0], 1), 0)
        time_data.insert(0, 'cases', tmp_array)
        for i, row in time_data.iterrows():
            if row['ambulanceLocation__first__dispatchingEtaTs'] is pd.NaT:
                time_data.loc[i, 'ambulanceLocation__first__dispatchingEtaTs'] = time_data.loc[i, 'dispatchingTs']
        time_data.set_index('ambulanceLocation__first__dispatchingEtaTs', inplace=True, drop=True)
        hour_rate = time_data.resample(f'{data_frequency}min').count().loc[datetime.datetime(2020, 8, 31):]
        end_date = pd.to_datetime(hour_rate.index.values[-1])
        x = self.load_weather(end_date).to(dtype=dtype)
        y = torch.as_tensor(hour_rate['cases'].values).to(dtype=dtype)
        return TensorDataset(x, y)

    def load_weather(self, end_date):
        historical_weather = HistoricalWeather()

        start_date = datetime.datetime(2020, 8, 31)
        data = historical_weather.get_temperature(start_date, end_date)
        data.fillna(0, inplace=True)
        selected_data = data[['temp', 'rhum', 'pres']]
        selected_data.insert(0, 'hour', selected_data.index.hour)
        selected_data.insert(1, 'day in month', selected_data.index.day)
        selected_data.insert(2, 'month', selected_data.index.month)

        football = self.load_football(end_date)
        selected_data.insert(6, 'football', football.values)

        self.scaler.fit(selected_data.values)
        transformed_values = self.scaler.transform(selected_data.values)
        return torch.as_tensor(transformed_values)

    def load_football(self, end_date):
        football = Football(self.teams)

        hourly_visitors = football.get_visitors()
        df = pd.DataFrame.from_dict(hourly_visitors, orient='index')
        return df[:end_date]

    def get_min_max(self):
        return self.dset[:][0].min(dim=0)[0].tolist(), self.dset[:][0].max(dim=0)[0].tolist()

    def inverse_transform(self, X: torch.Tensor):
        return self.scaler.inverse_transform(X.numpy())


def load_data(data_path):
    data = pd.read_json(data_path, lines=True)
    return data


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
    data_ = load_data(hack_config.z_case)
    data_loader = DataFactory(data_, hack_config.data_frequency, teams=hack_config.teams)
    print(data_loader.get_min_max())
