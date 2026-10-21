import json
import os

import streamlit as st
from google import genai

from weather_service import get_weather


st.set_page_config(
    page_title="ZIP Code Weather Agent",
    page_icon="🌦️",
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
MODEL_NAME = read_setting("GEMINI_MODEL", "gemini-3.8-flash")
APP_PASSWORD = read_setting("APP_PASSWORD")


st.title("🌦️ ZIP Code Weather Agent")
st.caption("Ask for current weather or a three-day forecast.")


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


client = genai.Client(api_key=API_KEY)


# This is the function description Gemini sees.
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": (
        "Get the current weather and a three-day forecast for a "
        "ZIP or postal code. Use this whenever the user asks for "
        "real weather information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "postal_code": {
                "type": "string",
                "description": (
                    "The ZIP or postal code, for example 45202."
                ),
            },
            "country_code": {
                "type": "string",
                "description": (
                    "Two-letter country code such as US, CA, GB "
                    "or IN. Use US when the user does not specify."
                ),
            },
            "unit": {
                "type": "string",
                "enum": ["fahrenheit", "celsius"],
                "description": (
                    "Temperature unit. Use fahrenheit by default "
                    "for US locations."
                ),
            },
        },
        "required": ["postal_code"],
    },
}


SYSTEM_INSTRUCTION = """
You are a helpful weather assistant.

Rules:
1. Use get_weather for every request involving actual current or
   forecast weather.
2. Never invent weather information.
3. If the user does not provide a ZIP or postal code, ask for it.
4. Default to country code US unless the user specifies another country.
5. Default to Fahrenheit for US locations and Celsius elsewhere.
6. Clearly mention the resolved city and state.
7. Explain API errors in friendly language.
8. The available forecast is limited to three days.
9. Keep responses concise and practical.
"""


def create_interaction(input_value, previous_interaction_id=None):
    """
    Send either a user message or a tool result to Gemini.
    """

    arguments = {
        "model": MODEL_NAME,
        "input": input_value,
        "system_instruction": SYSTEM_INSTRUCTION,
        "tools": [WEATHER_TOOL],
    }

    if previous_interaction_id:
        arguments["previous_interaction_id"] = (
            previous_interaction_id
        )

    return client.interactions.create(**arguments)


def run_weather_tool(arguments):
    """Execute the Python function requested by Gemini."""

    return get_weather(
        postal_code=arguments.get("postal_code", ""),
        country_code=arguments.get("country_code", "US"),
        unit=arguments.get("unit", "fahrenheit"),
    )


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


if st.sidebar.button("Start new conversation"):
    st.session_state.messages = []
    st.session_state.previous_interaction_id = None
    st.rerun()


with st.sidebar:
    st.markdown("### Example questions")
    st.markdown(
        """
- What is the weather in ZIP code 45202?
- Will it rain tomorrow in 10001?
- Give me the three-day forecast for 90210.
- What is the weather for postal code M5V 3A8 in Canada?
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