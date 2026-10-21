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

    return WEATHER_CODES.get(int(code), f"Unknown weather code: {code}")


def lookup_postal_code(postal_code: str, country_code: str = "US") -> dict:
    """
    Convert a postal/ZIP code into a location and coordinates.

    Example:
        lookup_postal_code("45202", "US")
    """

    postal_code = postal_code.strip()
    country_code = country_code.strip().upper()

    if not postal_code:
        return {
            "success": False,
            "error": "A postal or ZIP code is required.",
        }

    if len(country_code) != 2:
        return {
            "success": False,
            "error": "Use a two-letter country code such as US, CA, GB or IN.",
        }

    safe_country = quote(country_code.lower(), safe="")
    safe_postal_code = quote(postal_code, safe="")

    url = (
        f"https://api.zippopotam.us/"
        f"{safe_country}/{safe_postal_code}"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 404:
            return {
                "success": False,
                "error": (
                    f"Postal code {postal_code} was not found "
                    f"for country {country_code}."
                ),
            }

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        return {
            "success": False,
            "error": f"Postal-code service failed: {error}",
        }

    places = data.get("places", [])

    if not places:
        return {
            "success": False,
            "error": "The postal-code service returned no locations.",
        }

    place = places[0]

    try:
        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
    except (KeyError, TypeError, ValueError):
        return {
            "success": False,
            "error": "The location did not contain valid coordinates.",
        }

    return {
        "success": True,
        "postal_code": data.get("post code", postal_code),
        "country": data.get("country"),
        "country_code": data.get("country abbreviation", country_code),
        "place_name": place.get("place name"),
        "state": place.get("state"),
        "latitude": latitude,
        "longitude": longitude,
    }


def get_weather(
    postal_code: str,
    country_code: str = "US",
    unit: str = "fahrenheit",
) -> dict:
    """
    Get current weather and a three-day forecast for a postal code.

    Args:
        postal_code: ZIP or postal code, such as 45202.
        country_code: Two-letter country code; defaults to US.
        unit: Either fahrenheit or celsius.
    """

    unit = unit.strip().lower()

    if unit not in {"fahrenheit", "celsius"}:
        return {
            "success": False,
            "error": "Unit must be fahrenheit or celsius.",
        }

    # First convert the postal code into latitude and longitude.
    location = lookup_postal_code(postal_code, country_code)

    if not location["success"]:
        return location

    if unit == "fahrenheit":
        temperature_unit = "fahrenheit"
        wind_speed_unit = "mph"
        precipitation_unit = "inch"
    else:
        temperature_unit = "celsius"
        wind_speed_unit = "kmh"
        precipitation_unit = "mm"

    weather_url = "https://api.open-meteo.com/v1/forecast"

    parameters = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": (
            "temperature_2m,"
            "apparent_temperature,"
            "relative_humidity_2m,"
            "precipitation,"
            "weather_code,"
            "wind_speed_10m"
        ),
        "daily": (
            "weather_code,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_probability_max"
        ),
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
        "precipitation_unit": precipitation_unit,
        "timezone": "auto",
        "forecast_days": 3,
    }

    try:
        response = requests.get(
            weather_url,
            params=parameters,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        return {
            "success": False,
            "error": f"Weather service failed: {error}",
        }

    current = data.get("current", {})
    current_units = data.get("current_units", {})
    daily = data.get("daily", {})
    daily_units = data.get("daily_units", {})

    forecast = []

    forecast_dates = daily.get("time", [])

    for index, forecast_date in enumerate(forecast_dates):
        forecast.append(
            {
                "date": forecast_date,
                "condition": weather_description(
                    daily.get("weather_code", [None])[index]
                ),
                "high": daily.get(
                    "temperature_2m_max", [None]
                )[index],
                "low": daily.get(
                    "temperature_2m_min", [None]
                )[index],
                "precipitation_probability": daily.get(
                    "precipitation_probability_max", [None]
                )[index],
            }
        )

    location_name = ", ".join(
        part
        for part in [
            location.get("place_name"),
            location.get("state"),
            location.get("country"),
        ]
        if part
    )

    return {
        "success": True,
        "location": location_name,
        "postal_code": location["postal_code"],
        "coordinates": {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
        },
        "timezone": data.get("timezone"),
        "current": {
            "time": current.get("time"),
            "condition": weather_description(
                current.get("weather_code")
            ),
            "temperature": current.get("temperature_2m"),
            "temperature_unit": current_units.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "humidity_unit": current_units.get(
                "relative_humidity_2m"
            ),
            "precipitation": current.get("precipitation"),
            "precipitation_unit": current_units.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_speed_unit": current_units.get("wind_speed_10m"),
        },
        "forecast_units": {
            "temperature": daily_units.get("temperature_2m_max"),
            "precipitation_probability": daily_units.get(
                "precipitation_probability_max"
            ),
        },
        "three_day_forecast": forecast,
    }


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_weather("45202", "US", "fahrenheit"))