
import os
import time
from google import genai
from google.genai import types

# ----------------------------------
# 1. Initialize Gemini
# ----------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing! Set it in your terminal."
    )

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=120000
    )
)

# ----------------------------------
# 2. Define our custom tool
# ----------------------------------

def metro_fare_estimator(stations: int) -> str:
    """
    Calculate metro fare based on number of stops.

    Args:
        stations: Number of metro stops.
    """

    fare = min(10 + (stations * 4), 60)

    return (
        f"Your metro ride for {stations} stops "
        f"costs ₹{fare}!"
    )


# ----------------------------------
# 3. Create the AI agent
# ----------------------------------

print("Starting Gemini 3.6 Flash agent...")

chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        tools=[metro_fare_estimator]
    )
)

# ----------------------------------
# 4. Send a message to the agent
# ----------------------------------

question = (
    "I am riding the metro for 8 stops. "
    "Use the metro fare calculator to calculate "
    "my fare and explain the result."
)

# ----------------------------------
# 5. Get response with retry handling
# ----------------------------------

max_attempts = 3

for attempt in range(max_attempts):

    try:
        print("\nCalling Gemini...")

        response = chat.send_message(question)

        print("\n========== AGENT RESPONSE ==========")
        print(response.text)
        print("====================================")

        break

    except Exception as e:

        error = str(e)

        print(f"\nAttempt {attempt + 1} failed:")
        print(error)

        if "503" in error or "504" in error:

            if attempt < max_attempts - 1:
                wait_time = 5 * (2 ** attempt)

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                print(
                    "\nGemini is currently unavailable. "
                    "Please try again later."
                )

        else:
            print("\nUnexpected error. Check the API configuration.")
            break
