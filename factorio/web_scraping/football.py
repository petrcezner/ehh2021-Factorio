import requests
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup


class Football:
    def __init__(self,
                 teams_dct,
                 capacity_ratio=0.75,
                 hours_per_match=3):

        self.__teams = teams_dct["teams"]
        self.__capacity_ratio = capacity_ratio
        self.__hours_per_match = hours_per_match

    def get_visitors(self,
                     start_date=datetime(2020, 8, 31),
                     end_date=datetime(2021, 11, 18, 23, 59)):

        # get all matches
        matches = {}
        for i in range(start_date.year + 1, end_date.year + 2):
            new_matches = self.__get_visitors_date(str(i))
            matches.update(new_matches)

        hourly_visitors = {}
        for single_date in self.__daterange(start_date, end_date):
            if single_date in matches:
                hourly_visitors[single_date] = matches[single_date]
            else:
                hourly_visitors[single_date] = 0

        return hourly_visitors

    def __get_visitors_date(self, year):
        URL = "https://www.fortunaliga.cz/rozpis-zapasu/" + year + "?type=2&id_stage=1&round=0"
        page = requests.get(URL)

        soup = BeautifulSoup(page.content, "html.parser")

        job_elements = soup.find_all("tr", {"class": "game"})

        events = {}
        for job_element in job_elements:
            date_element = job_element.find("td", class_="date")
            date = date_element.find("span", class_="hidden-sm hidden-md hidden-lg").text
            # split to date
            x = date.split(", ")
            date = x[0].split("/")
            day = int(date[0])
            month = int(date[1])
            year = int("20" + date[2])

            time = x[1].split(":")
            start_hour = int(time[0])
            start_minute = int(time[1])

            club_element = job_element.find("td", class_="team home")
            club = club_element.find("span", class_="hidden-xs").text

            for team in self.__teams:
                if team["name"] == club:
                    capacity = team["capacity"]
                    for i in range(0, self.__hours_per_match):
                        actual_hour = start_hour + i
                        events[datetime(year, month, day, actual_hour, start_minute)] = int(
                            capacity * self.__capacity_ratio)

        return events

    def __daterange(self, start_date, end_date):
        delta = timedelta(hours=1)
        while start_date < end_date:
            yield start_date
            start_date += delta


if __name__ == '__main__':
    config_file = 'config.json'

    with open(config_file) as json_file:
        c = json.load(json_file)

    football = Football(c)

    hourly_visitors = football.get_visitors(start_date=datetime(2020, 11, 19),
                                            end_date=datetime(2021, 11, 21))

    for date, visitiors in hourly_visitors.items():
        print(str(date) + " | " + str(visitiors))
