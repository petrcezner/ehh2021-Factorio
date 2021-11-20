import pandas as pd
from datetime import datetime, timedelta


class MobilityWaze:
    def __init__(self,
                 datafile="Waze _ COVID-19 Impact Dashboard_City-Level Data_Table.csv"):
        self.__reports_df = pd.read_csv(datafile)

    def get_mobility(self,
                     start_date=datetime(2020, 8, 31),
                     end_date=datetime(2021, 11, 18, 23, 59)):
        mobility = self.__reports_df.loc[self.__reports_df['Country'] == "Prague"]

        mobility = mobility[['Date',
                             '% Change In Waze Driven Miles/KMs']]

        hourly_mobility = {}
        for index, mob in mobility.iterrows():
            time = mob['Date'].split(", ")
            moth_day = time[0].split(" ")
            year = int(time[1])
            datetime_object = datetime.strptime(moth_day[0], "%b")
            month = datetime_object.month
            day = int(moth_day[1])

            for hour in range(0, 24):
                date = datetime(year, month, day, hour, 0)
                if start_date < date < end_date:
                    hourly_mobility[date] = mob['% Change In Waze Driven Miles/KMs']

        return hourly_mobility


if __name__ == '__main__':
    mobility_waze = MobilityWaze()
    mobility = mobility_waze.get_mobility()

    for date, data in mobility.items():
        print(str(date) + " | " + str(data))
