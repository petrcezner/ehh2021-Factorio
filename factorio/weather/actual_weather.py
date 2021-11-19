import requests, json


# https://www.geeksforgeeks.org/python-find-current-weather-of-any-city-using-openweathermap-api/
# https://openweathermap.org/current
class ActualWeather:
    def __init__(self,
                 city="Prague",
                 api_key="3d007207f1014cf55bd0d79cacb9e194",
                 base_url="http://api.openweathermap.org/data/2.5/weather?"):
        self.__city = city
        self.__api_key = api_key
        self.__base_url = base_url

    def get_temperature(self):
        weather = self.__request_weather()

        return weather["temp"] - 273.15

    def get_pressure(self):
        weather = self.__request_weather()

        return weather["pressure"]

    def get_humidity(self):
        weather = self.__request_weather()

        return weather["humidity"]

    def __request_weather(self):
        complete_url = self.__base_url + "appid=" +self.__api_key + "&q=" + self.__city

        response = requests.get(complete_url)

        x = response.json()

        if x["cod"] != "404":
            return x["main"]
        else:
            return None


if __name__ == '__main__':
    actual_weather = ActualWeather()

    print(" Temperature (in celsius unit) = " +
          str(actual_weather.get_temperature()) +
          "\n atmospheric pressure (in hPa unit) = " +
          str(actual_weather.get_pressure()) +
          "\n humidity (in percentage) = " +
          str(actual_weather.get_humidity()))

