import requests

API_KEY = "ea02f8ae80ad1f3eded318b35f9b52b9"

def get_weather(city):

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    data = response.json()

    if data.get("cod") != 200:
        return None

    weather = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind": data["wind"]["speed"]
    }

    return weather