from urllib.parse import quote

import requests


# Open-Meteo returns WMO weather codes.
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Freezing fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Heavy drizzle",
    56: "Light freezing drizzle",
    57: "Heavy freezing drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Moderate rain showers",
    82: "Heavy rain showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with light hail",
    99: "Thunderstorm with heavy hail",
}


def weather_description(code):
    """Convert an Open-Meteo weather code into readable text."""
    if code is None:
        return "Unknown"

    return WEATHER_CODES.get(int(code), "Unknown weather condition")


def get_value(data, field, index):
    """Safely retrieve an item from an Open-Meteo result array."""
    values = data.get(field, [])

    if index < len(values):
        return values[index]

    return None


def get_weather(
    zip_code: str,
    forecast_type: str = "current",
    days: int = 7,
    hours: int = 24
) -> dict:
    """
    Get current, hourly or daily weather for a US ZIP code.

    Args:
        zip_code: Five-digit US ZIP code.
        forecast_type: current, hourly, daily or all.
        days: Number of daily forecast days from 1 through 10.
        hours: Number of hourly forecast hours from 1 through 48.

    Returns:
        A dictionary containing location and weather information.
    """

    zip_code = str(zip_code).strip()
    forecast_type = forecast_type.lower().strip()

    if len(zip_code) != 5 or not zip_code.isdigit():
        return {
            "error": "Please provide a valid five-digit US ZIP code."
        }

    allowed_types = {"current", "hourly", "daily", "all"}

    if forecast_type not in allowed_types:
        return {
            "error": (
                "forecast_type must be current, hourly, daily or all."
            )
        }

    # Prevent excessively large API and Gemini responses.
    days = max(1, min(int(days), 10))
    hours = max(1, min(int(hours), 48))

    try:
        # Step 1: Convert ZIP code into coordinates.
        location_url = (
            f"https://api.zippopotam.us/us/{zip_code}"
        )

        location_response = requests.get(
            location_url,
            timeout=10
        )

        if location_response.status_code == 404:
            return {
                "error": f"ZIP code {zip_code} was not found."
            }

        location_response.raise_for_status()
        location_data = location_response.json()

        place = location_data["places"][0]

        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
        city = place["place name"]
        state = place["state abbreviation"]

        # Step 2: Build the Open-Meteo request.
        weather_url = "https://api.open-meteo.com/v1/forecast"

        parameters = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
        }

        if forecast_type in {"current", "all"}:
            parameters["current"] = (
                "temperature_2m,"
                "apparent_temperature,"
                "relative_humidity_2m,"
                "precipitation,"
                "weather_code,"
                "wind_speed_10m,"
                "wind_gusts_10m"
            )

        if forecast_type in {"hourly", "all"}:
            parameters["hourly"] = (
                "temperature_2m,"
                "apparent_temperature,"
                "precipitation_probability,"
                "weather_code,"
                "wind_speed_10m"
            )

            # Start at the current hour.
            parameters["forecast_hours"] = hours

        if forecast_type in {"daily", "all"}:
            parameters["daily"] = (
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max,"
                "wind_speed_10m_max,"
                "sunrise,"
                "sunset"
            )

            parameters["forecast_days"] = days

        weather_response = requests.get(
            weather_url,
            params=parameters,
            timeout=15
        )

        weather_response.raise_for_status()
        weather_data = weather_response.json()

        result = {
            "location": {
                "zip_code": zip_code,
                "city": city,
                "state": state,
                "latitude": latitude,
                "longitude": longitude,
                "timezone": weather_data.get("timezone"),
            },
            "requested_forecast_type": forecast_type,
        }

        # Step 3: Format current weather.
        if "current" in weather_data:
            current = weather_data["current"]

            result["current"] = {
                "time": current.get("time"),
                "condition": weather_description(
                    current.get("weather_code")
                ),
                "temperature_f": current.get("temperature_2m"),
                "feels_like_f": current.get(
                    "apparent_temperature"
                ),
                "humidity_percent": current.get(
                    "relative_humidity_2m"
                ),
                "precipitation_inches": current.get(
                    "precipitation"
                ),
                "wind_mph": current.get("wind_speed_10m"),
                "wind_gust_mph": current.get(
                    "wind_gusts_10m"
                ),
            }

        # Step 4: Format hourly forecast.
        if "hourly" in weather_data:
            hourly_data = weather_data["hourly"]
            times = hourly_data.get("time", [])
            hourly_forecast = []

            for index, forecast_time in enumerate(times):
                weather_code = get_value(
                    hourly_data,
                    "weather_code",
                    index
                )

                hourly_forecast.append({
                    "time": forecast_time,
                    "condition": weather_description(
                        weather_code
                    ),
                    "temperature_f": get_value(
                        hourly_data,
                        "temperature_2m",
                        index
                    ),
                    "feels_like_f": get_value(
                        hourly_data,
                        "apparent_temperature",
                        index
                    ),
                    "precipitation_probability_percent":
                        get_value(
                            hourly_data,
                            "precipitation_probability",
                            index
                        ),
                    "wind_mph": get_value(
                        hourly_data,
                        "wind_speed_10m",
                        index
                    ),
                })

            result["hourly_forecast"] = hourly_forecast

        # Step 5: Format daily forecast.
        if "daily" in weather_data:
            daily_data = weather_data["daily"]
            dates = daily_data.get("time", [])
            daily_forecast = []

            for index, forecast_date in enumerate(dates):
                weather_code = get_value(
                    daily_data,
                    "weather_code",
                    index
                )

                daily_forecast.append({
                    "date": forecast_date,
                    "condition": weather_description(
                        weather_code
                    ),
                    "high_f": get_value(
                        daily_data,
                        "temperature_2m_max",
                        index
                    ),
                    "low_f": get_value(
                        daily_data,
                        "temperature_2m_min",
                        index
                    ),
                    "precipitation_probability_percent":
                        get_value(
                            daily_data,
                            "precipitation_probability_max",
                            index
                        ),
                    "maximum_wind_mph": get_value(
                        daily_data,
                        "wind_speed_10m_max",
                        index
                    ),
                    "sunrise": get_value(
                        daily_data,
                        "sunrise",
                        index
                    ),
                    "sunset": get_value(
                        daily_data,
                        "sunset",
                        index
                    ),
                })

            result["daily_forecast"] = daily_forecast

        return result

    except requests.RequestException as error:
        return {
            "error": f"Weather service request failed: {error}"
        }

    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {
            "error": f"Weather data could not be processed: {error}"
        }