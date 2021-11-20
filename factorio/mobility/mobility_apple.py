import pandas as pd
from datetime import datetime, timedelta


class MobilityApple:
    def __init__(self,
                 datafile="applemobilitytrends-2021-11-18.csv"):
        self.__reports_df = pd.read_csv(datafile)

    def get_mobility(self,
                     start_date=datetime(2020, 8, 31),
                     end_date=datetime(2021, 11, 18, 23, 59)):
        mobility = self.__reports_df.loc[self.__reports_df['region'] == "Prague"]
        dates = mobility.drop(['geo_type',
                               'region',
                               'transportation_type',
                               'alternative_name',
                               'sub-region',
                               'country'], axis=1)

        hourly_mobility = {}
        for index, record in dates.iterrows():
            for i, v in record.items():
                time = i.split("-")
                year = int(time[0])
                month = int(time[1])
                day = int(time[2])

                for hour in range(0,24):
                    date = datetime(year, month, day, hour, 0)
                    if start_date < date < end_date:
                        hourly_mobility[date] = v

        return hourly_mobility


if __name__ == '__main__':
    mobility_apple = MobilityApple()
    mobility = mobility_apple.get_mobility()

    for date, data in mobility.items():
        print(str(date) + " | " + str(data))

