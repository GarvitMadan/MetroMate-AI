
import os
import time
from google import genai
from google.genai import types

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing!")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=120000)
)


def metro_fare_estimator(stations: int) -> str:
    fare = min(10 + (stations * 4), 60)
    return f"Your metro ride for {stations} stops costs ₹{fare}!"


print("Starting MetroMate AI...")

chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        tools=[metro_fare_estimator]
    )
)

question = (
    "I am riding the metro for 8 stops. "
    "Use the metro fare calculator to calculate my fare."
)

max_attempts = 3

for attempt in range(max_attempts):
    try:
        print("Calling Gemini...")

        response = chat.send_message(question)

        print("\nMetroMate AI Response:\n")
        print(response.text)
        break

    except Exception as e:
        print(f"\nAttempt {attempt + 1} failed: {e}")

        if ("503" in str(e) or "504" in str(e)) and attempt < max_attempts - 1:
            wait_time = 5 * (2 ** attempt)
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print("Unable to complete the request.")
            break
