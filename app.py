import json
import os

import streamlit as st
from google import genai
from google.genai import types

from weather_service import get_weather


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
        color: white;
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
MODEL_NAME = read_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")
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
    http_options=types.HttpOptions(timeout=30_000),
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


def create_interaction(input_value, previous_interaction_id=None):
    """
    Send either a user message or a tool result to Gemini.
    """

    arguments = {
        "model": MODEL_NAME,
        "input": input_value,
        "system_instruction": SYSTEM_INSTRUCTION,
        "tools": [weather_tool],
    }

    if previous_interaction_id:
        arguments["previous_interaction_id"] = (
            previous_interaction_id
        )

    return client.interactions.create(**arguments)


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
    st.session_state.weather_data_used = True
    print("TOOL RESULT:", result)
     
    return result


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
        st.markdown(message["content"])


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
        with st.spinner("Checking the weather..."):
            try:
                answer, interaction_id = ask_agent(
                    user_message,
                    st.session_state.previous_interaction_id,
                )

                st.session_state.previous_interaction_id = (
                    interaction_id
                )

            except Exception as error:
                answer = (
                    "I could not complete the request. "
                    f"Technical details: {error}"
                )

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
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
            AI responses generated with Google Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )