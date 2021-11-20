import pandas as pd
from datetime import datetime, timedelta


class MobilityGoogle:
    def __init__(self,
                 reports=['2020_CZ_Region_Mobility_Report.csv',
                          '2021_CZ_Region_Mobility_Report.csv']):

        df = None
        for count, report in enumerate(reports):
            if count == 0:
                df = pd.read_csv(reports[count])
            else:
                df_2 = pd.read_csv(reports[count])
                data = [df, df_2]
                df = pd.concat(data)

        self.__reports_df = df

    def get_df(self):
        df = self.__reports_df.loc[self.__reports_df['sub_region_1'] == "Prague"]

        df1 = df[['date',
                  'retail_and_recreation_percent_change_from_baseline',
                  'grocery_and_pharmacy_percent_change_from_baseline',
                  'parks_percent_change_from_baseline',
                  'transit_stations_percent_change_from_baseline',
                  'workplaces_percent_change_from_baseline',
                  'residential_percent_change_from_baseline']]

        return df1

    def get_mobility(self,
                     start_date=datetime(2020, 8, 31),
                     end_date=datetime(2021, 11, 18, 23, 59)):

        mobility = self.get_df()

        hourly_mobility = {}
        for index, mob in mobility.iterrows():
            time = mob['date'].split("-")
            year = int(time[0])
            month = int(time[1])
            day = int(time[2])

            for hour in range(0,24):
                date = datetime(year, month, day, hour, 0)
                if start_date < date < end_date:
                    hourly_mobility[date] = mob.iloc[1:]

        return hourly_mobility


if __name__ == '__main__':
    mobility_google = MobilityGoogle()
    df = mobility_google.get_df()
    mobility = mobility_google.get_mobility()

    for date, data in mobility.items():
        print(str(date) + " | " + str(data))