import pandas as pd
from datetime import datetime, timedelta


class MobilityApple:
    def __init__(self,
                 datafile="applemobilitytrends-2021-11-18.csv"):
        self.__reports_df = pd.read_csv(datafile)

    def get_mobility(self,
                    start_date = datetime(2020, 8, 31),
                    end_date = datetime(2021, 11, 18, 23, 59)):
        mobility = self.__reports_df.loc[self.__reports_df['region'] == "Prague"]

        hourly_mobility = {}
        for single_date in self.__daterange(start_date, end_date):
            if single_date in mobility['date']:
                hourly_mobility[single_date] = mobility.iloc[1:]

        return hourly_mobility

