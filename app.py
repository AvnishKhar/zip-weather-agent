import json
import os
import re
import time

import streamlit as st
from google import genai
from google.genai import types

from weather_service import get_radar_url, get_weather


st.set_page_config(
    page_title="ZIP Code Weather Agent",
    page_icon="🌦️",
)

st.markdown(
    """
    <style>
    .weather-title {
        display: inline-flex;
        flex-direction: column;
        align-items: flex-end;
        margin-bottom: 1rem;
    }

    .weather-heading {
        color: var(--text-color);
        font-size: clamp(2.5rem, 5vw, 3.5rem);
        font-weight: 700;
        line-height: 1.15;
    }

    .weather-signature {
        color: #2196F3;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 0.15rem;
        padding-right: 0.1rem;
    }
    </style>

    <div class="weather-title">
        <div class="weather-heading">
            🌦️ ZIP Code Weather Agent
        </div>
        <div class="weather-signature">
          Created by AvKh
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
def read_setting(name, default=None):
    """Read a setting from Streamlit secrets or an environment variable."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


API_KEY = read_setting("GEMINI_API_KEY")
MODEL_NAME = read_setting("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_MODEL_NAME = read_setting(
    "GEMINI_FALLBACK_MODEL",
    "gemini-flash-lite-latest",
)

# Status codes that usually mean "busy, try again shortly".
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
APP_PASSWORD = read_setting("APP_PASSWORD")


st.caption(
    "Ask for current weather, an hourly forecast, "
    "or a 7–10-day forecast."
)


# Optional basic protection for a publicly shared application.
if APP_PASSWORD:
    entered_password = st.sidebar.text_input(
        "Shared password",
        type="password",
    )

    if entered_password != APP_PASSWORD:
        st.info("Enter the shared password in the sidebar.")
        st.stop()


if not API_KEY:
    st.error(
        "GEMINI_API_KEY is missing. Add it to "
        ".streamlit/secrets.toml."
    )
    st.stop()


client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=90_000),
)


# This is the function description Gemini sees.
weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": (
        "Gets current conditions, an hourly forecast, a 7-to-10-day "
        "daily forecast, or all forecasts for a five-digit US ZIP code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "zip_code": {
                "type": "string",
                "description": "Five-digit US ZIP code, such as 45208."
            },
            "forecast_type": {
                "type": "string",
                "enum": [
                    "current",
                    "hourly",
                    "daily",
                    "all"
                ],
                "description": (
                    "Use current for present conditions, hourly for "
                    "hour-by-hour conditions, daily for a multi-day "
                    "forecast, or all for every forecast type."
                )
            },
            "days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": (
                    "Number of daily forecast days. Use 7 unless the "
                    "user specifically asks for 10 days."
                )
            },
            "hours": {
                "type": "integer",
                "minimum": 1,
                "maximum": 48,
                "description": (
                    "Number of hourly forecast hours. Use 24 unless "
                    "the user asks for another period."
                )
            }
        },
        "required": [
            "zip_code",
            "forecast_type"
        ]
    }
}


SYSTEM_INSTRUCTION = """
You are a conversational weather assistant.

Use the get_weather tool whenever the user asks about current or
future weather for a US ZIP code.

Choose forecast_type as follows:

- Use "current" for current, now, today, or present conditions.
- Use "hourly" for hourly, later today, tonight, or the next few hours.
- Use "daily" for a 7-day, weekly, 10-day, or extended forecast.
- Use "all" when the user asks for both hourly and daily forecasts.

Use 24 hours for an hourly forecast unless the user asks for a
different period. Never request more than 48 hours.

Use 7 days for a weekly forecast and 10 days for a 10-day forecast.

Clearly show the local date and time.
Do not invent weather data.
Mention that forecasts farther into the future have greater uncertainty.

Format the Current Conditions section without ordinary bullet points.
Start each item on a separate line using these icons:

🌡️ **Temperature:** value and feels-like temperature
☀️ **Condition:** description; choose an appropriate weather icon
💧 **Humidity:** value
💨 **Wind:** speed and gusts

Do not place hyphens or bullet symbols before the icons.
Use two trailing spaces after each line so Markdown displays each
weather condition on a separate line.
"""


def is_retryable_error(error):
    """Return True when Gemini is temporarily overloaded or unavailable."""

    status_code = getattr(error, "status_code", None)

    if status_code is None:
        status_code = getattr(error, "code", None)

    return status_code in RETRYABLE_STATUS_CODES


def create_interaction(input_value, previous_interaction_id=None):
    """
    Send either a user message or a tool result to Gemini.

    Retries temporary errors (such as 503 high demand) with backoff,
    then tries the fallback model before giving up.
    """

    arguments = {
        "input": input_value,
        "system_instruction": SYSTEM_INSTRUCTION,
        "tools": [weather_tool],
    }

    if previous_interaction_id:
        arguments["previous_interaction_id"] = (
            previous_interaction_id
        )

    models = [MODEL_NAME]

    if FALLBACK_MODEL_NAME and FALLBACK_MODEL_NAME != MODEL_NAME:
        models.append(FALLBACK_MODEL_NAME)

    last_error = None

    for model in models:
        for attempt in range(2):
            try:
                return client.interactions.create(
                    model=model,
                    **arguments,
                )
            except Exception as error:
                if not is_retryable_error(error):
                    raise

                last_error = error
                print(
                    f"GEMINI BUSY ({model}, attempt {attempt + 1}):",
                    repr(error),
                    flush=True,
                )

                # A rate limit (429) will not clear in a second or two,
                # and retrying only uses more quota, so move straight
                # on to the next model.
                if getattr(error, "status_code", None) == 429:
                    break

                if attempt == 0:
                    time.sleep(1)

    raise last_error


def extract_zip_code(text):
    """Return the first five-digit ZIP code in the text, if any."""

    match = re.search(r"\b\d{5}\b", text)

    return match.group(0) if match else None


# Words that need Gemini's judgment rather than a plain lookup.
COMPLEX_WORDS = [
    "tomorrow", "weekend", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "rain", "snow",
    "storm", "umbrella", "jacket", "coat", "should", "will",
    "compare", "versus", " vs", "best", "why",
]

WEATHER_WORDS = [
    "weather", "temperature", "temp", "conditions", "current",
    "now", "forecast", "hourly", "hours", "tonight", "daily",
    "week", "day", "extended",
]


def parse_simple_request(text):
    """
    Turn a simple weather request into get_weather arguments,
    or return None when Gemini should handle it.
    """

    zip_codes = re.findall(r"\b\d{5}\b", text)

    # Exactly one ZIP code keeps the request unambiguous.
    if len(zip_codes) != 1:
        return None

    zip_code = zip_codes[0]
    lowered = text.lower()

    if lowered.strip() == zip_code:
        return {"zip_code": zip_code, "forecast_type": "current"}

    if any(word in lowered for word in COMPLEX_WORDS):
        return None

    if not any(word in lowered for word in WEATHER_WORDS):
        return None

    wants_hourly = any(
        word in lowered
        for word in ["hourly", "hour", "tonight", "later today"]
    )
    # Match "day" as a whole word so "today" does not count.
    wants_daily = bool(
        re.search(r"\bdays?\b|daily|week|extended", lowered)
    )

    arguments = {"zip_code": zip_code, "forecast_type": "current"}

    if wants_hourly and wants_daily:
        arguments["forecast_type"] = "all"
    elif wants_hourly:
        arguments["forecast_type"] = "hourly"
    elif wants_daily:
        arguments["forecast_type"] = "daily"
    elif "forecast" in lowered:
        arguments["forecast_type"] = "daily"

    hours_match = re.search(r"(\d{1,2})\s*(?:-\s*)?hours?", lowered)

    if hours_match:
        arguments["hours"] = min(max(int(hours_match.group(1)), 1), 48)

    if re.search(r"\b(10|ten)[\s-]*days?\b", lowered):
        arguments["days"] = 10

    return arguments


def direct_weather_answer(arguments):
    """Look up the weather without going through Gemini."""

    try:
        result = run_weather_tool(arguments)
        return format_weather_response(result)
    except Exception as error:
        print("WEATHER TOOL ERROR:", repr(error), flush=True)
        return (
            "The weather service is not responding "
            "right now. Please try again in a moment."
        )


def run_weather_tool(arguments):
    """Execute the Python function requested by Gemini."""

    print("TOOL ARGUMENTS:", arguments)

    result = get_weather(
        zip_code=arguments.get("zip_code", ""),
        forecast_type=arguments.get(
            "forecast_type",
            "current",
        ),
        days=arguments.get("days", 7),
        hours=arguments.get("hours", 24),
    )

     # This line runs only after get_weather completes successfully.
    if "error" not in result:
     st.session_state.weather_data_used = True
     # Remember the location so the radar can be shown after the answer.
     st.session_state.last_location = result.get("location")
    print("TOOL RESULT:", result)

    return result


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_radar_url(latitude, longitude):
    """The nearest radar station never changes, so cache it per location."""

    return get_radar_url(latitude, longitude)


RADAR_CAPTION = "Live radar loop from the nearest NWS radar station"


# Animated Meteocons icons (MIT license), loaded by the browser from a CDN.
ICON_BASE_URL = (
    "https://cdn.jsdelivr.net/gh/basmilius/weather-icons@2.0.0/"
    "production/fill/all/"
)


def weather_category(condition):
    """Group a condition description into a simple weather category."""

    text = str(condition).lower()

    if "thunder" in text:
        return "thunder"
    if "hail" in text:
        return "hail"
    if "freezing" in text:
        return "sleet"
    if "snow" in text:
        return "snow"
    if "drizzle" in text:
        return "drizzle"
    if any(word in text for word in ["rain", "shower"]):
        return "rain"
    if "fog" in text:
        return "fog"
    if "overcast" in text:
        return "overcast"
    if "partly" in text:
        return "partly"
    if "cloud" in text:
        return "cloudy"

    return "clear"


ICON_NAMES = {
    "thunder": "thunderstorms-rain",
    "hail": "hail",
    "sleet": "sleet",
    "snow": "snow",
    "drizzle": "drizzle",
    "rain": "rain",
    "fog": "fog",
    "overcast": "overcast",
    "partly": "partly-cloudy-day",
    "cloudy": "cloudy",
    "clear": "clear-day",
}

NIGHT_ICON_NAMES = {
    "partly": "partly-cloudy-night",
    "clear": "clear-night",
}

# Translucent gradients read well in both light and dark themes.
ROW_BACKGROUNDS = {
    "thunder": "linear-gradient(90deg, rgba(91,33,182,.45), rgba(30,41,59,.25))",
    "hail": "linear-gradient(90deg, rgba(148,163,184,.40), rgba(59,130,246,.15))",
    "sleet": "linear-gradient(90deg, rgba(148,163,184,.35), rgba(186,230,253,.20))",
    "snow": "linear-gradient(90deg, rgba(224,242,254,.45), rgba(186,230,253,.15))",
    "drizzle": "linear-gradient(90deg, rgba(96,165,250,.30), rgba(148,163,184,.15))",
    "rain": "linear-gradient(90deg, rgba(37,99,235,.40), rgba(71,85,105,.20))",
    "fog": "linear-gradient(90deg, rgba(203,213,225,.35), rgba(148,163,184,.10))",
    "overcast": "linear-gradient(90deg, rgba(100,116,139,.40), rgba(148,163,184,.15))",
    "partly": "linear-gradient(90deg, rgba(251,191,36,.30), rgba(148,163,184,.20))",
    "cloudy": "linear-gradient(90deg, rgba(148,163,184,.35), rgba(203,213,225,.15))",
    "clear": "linear-gradient(90deg, rgba(251,191,36,.40), rgba(56,189,248,.20))",
}

NIGHT_ROW_BACKGROUNDS = {
    "partly": "linear-gradient(90deg, rgba(30,58,138,.50), rgba(100,116,139,.20))",
    "clear": "linear-gradient(90deg, rgba(30,27,75,.65), rgba(30,58,138,.30))",
}


def weather_row(content, condition, is_day=True):
    """Wrap one forecast line in a weather-themed background."""

    category = weather_category(condition)
    background = ROW_BACKGROUNDS[category]

    if not is_day:
        background = NIGHT_ROW_BACKGROUNDS.get(category, background)

    return (
        f'<div style="background:{background};border-radius:10px;'
        f'padding:0.35rem 0.75rem;margin:0.3rem 0;">{content}</div>'
    )


def weather_icon(condition, is_day=True):
    """Return an animated icon that matches the weather condition."""

    category = weather_category(condition)
    name = ICON_NAMES[category]

    if not is_day:
        name = NIGHT_ICON_NAMES.get(category, name)

    return (
        f'<img src="{ICON_BASE_URL}{name}.svg" alt="" '
        'style="height:1.6em;vertical-align:middle;">'
    )


def display_value(value, suffix=""):
    """Display N/A when a weather value is missing."""

    if value is None:
        return "N/A"

    return f"{value}{suffix}"


def format_weather_response(result):
    """Convert the weather dictionary into Streamlit Markdown."""

    if not isinstance(result, dict):
        return "I could not process the weather response."

    if result.get("error"):
        return f"⚠️ {result['error']}"

    location = result.get("location", {})
    city = location.get("city", "")
    state = location.get("state", "")
    zip_code = location.get("zip_code", "")

    location_name = ", ".join(
        value for value in [city, state] if value
    )

    heading = f"### Weather for {location_name}"

    if zip_code:
        heading += f" ({zip_code})"

    sections = [heading]

    current = result.get("current")

    if current:
        condition = current.get("condition", "Unknown")

        sections.append(
            "#### Current Conditions\n\n"
            f"🕒 **Local time:** "
            f"{current.get('time', 'N/A')}  \n"
            f"🌡️ **Temperature:** "
            f"{display_value(current.get('temperature_f'), '°F')} "
            f"(feels like "
            f"{display_value(current.get('feels_like_f'), '°F')})  \n"
            f"{weather_icon(condition)} **Condition:** "
            f"{condition}  \n"
            f"💧 **Humidity:** "
            f"{display_value(current.get('humidity_percent'), '%')}  \n"
            f"💨 **Wind:** "
            f"{display_value(current.get('wind_mph'), ' mph')} "
            f"(gusts "
            f"{display_value(current.get('wind_gust_mph'), ' mph')})"
        )

    hourly = result.get("hourly_forecast", [])

    if hourly:
        lines = []

        for item in hourly:
            condition = item.get("condition", "Unknown")
            is_day = item.get("is_day", True)

            # Rows are raw HTML, so use <b> rather than Markdown **.
            lines.append(
                weather_row(
                    f"<b>{item.get('time', 'N/A')}</b> — "
                    f"{weather_icon(condition, is_day)} {condition}, "
                    f"🌡️ {display_value(item.get('temperature_f'), '°F')}, "
                    f"rain "
                    f"{display_value(item.get('precipitation_probability_percent'), '%')}, "
                    f"💨 {display_value(item.get('wind_mph'), ' mph')}"
                    + (
                        f" {item['wind_direction']}"
                        if item.get("wind_direction")
                        else ""
                    ),
                    condition,
                    is_day,
                )
            )

        sections.append(
            "#### Hourly Forecast\n\n" + "\n".join(lines)
        )

    daily = result.get("daily_forecast", [])

    if daily:
        lines = []

        for item in daily:
            condition = item.get("condition", "Unknown")

            lines.append(
                weather_row(
                    f"<b>{item.get('date', 'N/A')}</b> — "
                    f"{weather_icon(condition)} {condition}; "
                    f"high "
                    f"{display_value(item.get('high_f'), '°F')}, "
                    f"low "
                    f"{display_value(item.get('low_f'), '°F')}; "
                    f"rain "
                    f"{display_value(item.get('precipitation_probability_percent'), '%')}; "
                    f"maximum wind "
                    f"{display_value(item.get('maximum_wind_mph'), ' mph')}",
                    condition,
                )
            )

        sections.append(
            "#### Daily Forecast\n\n" + "\n".join(lines)
        )

    if hourly or daily:
        sections.append(
            "_Forecasts farther into the future have greater uncertainty._"
        )

    return "\n\n".join(sections)

def ask_agent(user_message, previous_interaction_id=None):
    """
    Send the message to Gemini, execute requested tools,
    and return Gemini's final response.
    """

    interaction = create_interaction(
        user_message,
        previous_interaction_id,
    )

    # Limit the loop so a model cannot request tools indefinitely.
    for _ in range(5):
        function_calls = [
            step
            for step in (interaction.steps or [])
            if getattr(step, "type", None) == "function_call"
        ]

        # No function call means Gemini has produced its final answer.
        if not function_calls:
            answer = interaction.output_text

            if not answer:
                answer = "I could not generate a response."

            return answer, interaction.id

        function_results = []

        for function_call in function_calls:
            arguments = dict(function_call.arguments or {})

            if function_call.name == "get_weather":
                try:
                    result = run_weather_tool(arguments)
                    return format_weather_response(result), None
                except Exception as error:
                   print(
                       "WEATHER TOOL ERROR:",
                       repr(error),
                       flush=True,
                   )

                   result = {
                       "success": False,
                        "error": f"Weather tool failed: {error}",
                   }     
            else:
                result = {
                    "success": False,
                    "error": (
                        f"Unknown tool requested: "
                        f"{function_call.name}"
                    ),
                }

            # Give the result back to Gemini.
            function_results.append(
                {
                    "type": "function_result",
                    "name": function_call.name,
                    "call_id": function_call.id,
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps(result),
                        }
                    ],
                }
            )

        interaction = create_interaction(
            function_results,
            interaction.id,
        )

    raise RuntimeError("The agent exceeded the tool-call limit.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "previous_interaction_id" not in st.session_state:
    st.session_state.previous_interaction_id = None

if "weather_data_used" not in st.session_state:
    st.session_state.weather_data_used = False

if st.sidebar.button("Start new conversation"):
    st.session_state.messages = []
    st.session_state.previous_interaction_id = None
    st.session_state.weather_data_used = False
    st.rerun()


with st.sidebar:
    st.markdown("### Example questions")
    st.markdown(
        """
- What is the weather in ZIP code 45202?
- Give me the next 12 hours for ZIP code 10001.
- Will it rain tomorrow in ZIP code 90210?
- Give me the 10-day forecast for ZIP code 33139.
        """
    )


# Re-display existing messages after every Streamlit rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            message["content"],
            unsafe_allow_html=message.get("html", False),
        )

        if message.get("radar_url"):
            st.image(message["radar_url"], caption=RADAR_CAPTION)


user_message = st.chat_input(
    "Example: What is the weather in 45202?"
)

if user_message:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        st.session_state.last_location = None

        with st.spinner("Checking the weather..."):
            try:
                # Simple requests need no AI, so skip Gemini and
                # save quota.
                simple_request = parse_simple_request(user_message)

                if simple_request:
                    answer = direct_weather_answer(simple_request)
                else:
                    answer, interaction_id = ask_agent(
                        user_message,
                        st.session_state.previous_interaction_id,
                    )

                    st.session_state.previous_interaction_id = (
                        interaction_id
                    )

            except Exception as error:
                print("AGENT ERROR:", repr(error), flush=True)

                zip_code = extract_zip_code(user_message)

                if is_retryable_error(error) and zip_code:
                    # Gemini is busy, but the weather data does not
                    # depend on it, so look the ZIP code up directly.
                    answer = direct_weather_answer(
                        {
                            "zip_code": zip_code,
                            "forecast_type": "current",
                        }
                    )
                elif is_retryable_error(error):
                    answer = (
                        "The AI service is busy right now. Please try "
                        "again in a moment, or include a five-digit "
                        "ZIP code to get current conditions directly."
                    )
                else:
                    answer = (
                        "I could not complete the request. "
                        f"Technical details: {error}"
                    )

        # Only our own weather formatter produces HTML (the animated
        # icons), and it runs only when weather data was fetched.
        location = st.session_state.last_location
        allow_html = location is not None

        st.markdown(answer, unsafe_allow_html=allow_html)

        # Look up the radar after the answer is already on screen,
        # so it never delays the weather text.
        radar_url = None

        if location and location.get("latitude") is not None:
            radar_url = cached_radar_url(
                location["latitude"],
                location["longitude"],
            )

        if radar_url:
            st.image(radar_url, caption=RADAR_CAPTION)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "html": allow_html,
            "radar_url": radar_url,
        }
    )
if st.session_state.weather_data_used:
    st.markdown(
        """
        <div style="
            text-align: center;
            color: #888888;
            font-size: 0.78rem;
            margin-top: 2rem;
        ">
            Weather data by
            <a href="https://open-meteo.com/" target="_blank">
                Open-Meteo
            </a>
            (<a href="https://creativecommons.org/licenses/by/4.0/"
                target="_blank">CC BY 4.0</a>;
            formatted for display) ·
            Radar by
            <a href="https://radar.weather.gov/" target="_blank">
                NOAA / National Weather Service
            </a>
            · Icons by
            <a href="https://github.com/basmilius/weather-icons"
                target="_blank">Meteocons</a> (MIT) ·
            AI responses generated with Google Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )