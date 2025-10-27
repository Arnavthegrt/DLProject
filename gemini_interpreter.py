import google.generativeai as genai
import json
import os

# Load the API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")

genai.configure(api_key=GOOGLE_API_KEY)

def get_travel_preferences(user_query):
    """
    Uses the Gemini API to understand a user's query AND recommend cities.
    """
    # Using the more powerful model is better for this reasoning task
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
    You are an expert travel assistant. Analyze the user query below.
    Your response MUST be a single, clean JSON object with two main keys: "preferences" and "recommendations".

    1.  The "preferences" object should contain the extracted travel details with the following keys:
        - "location_hint", "vibe", "interests", "budget", "duration_days", "dislikes".
        - If any preference is not mentioned, set its value to null.

    2.  The "recommendations" object should contain one key:
        - "recommended_cities": A JSON list of 3 to 5 city names (e.g., ["Lisbon, Portugal", "Crete, Greece"]) that are excellent matches for the user's preferences.

    User Query: "{user_query}"
    """

    try:
        response = model.generate_content(prompt)
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"Error processing Gemini response: {e}")
        return None

# --- Example Usage ---
if __name__ == '__main__':
    query = "I'm looking for a relaxing 7-day trip to a beach destination in Europe. I'm on a medium budget and love historical sites but not big crowds."
    result = get_travel_preferences(query)
    print(json.dumps(result, indent=2))