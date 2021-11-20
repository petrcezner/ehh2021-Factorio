import pandas as pd
from datetime import datetime, timedelta


class MobilityWaze:
    def __init__(self,
                 datafile="Waze _ COVID-19 Impact Dashboard_City-Level Data_Table.csv"):
        self.__reports_df = pd.read_csv(datafile)

    def get_mobility(self,
                    start_date = datetime(2020, 8, 31),
                    end_date = datetime(2021, 11, 18, 23, 59)):
        mobility = self.__reports_df.loc[self.__reports_df['Country'] == "Prague"]

        hourly_mobility = {}
        for index, mob in mobility.iterrows():
            time = mob['Date'].split(" ")
            year = int(time[0])
            month = int(time[1])
            day = int(time[2])

            for hour in range(0,24):
                date = datetime(year, month, day, hour, 0)
                if start_date < date < end_date:
                    hourly_mobility[date] = mob['% Change In Waze Driven Miles/KMs']

        return hourly_mobility
