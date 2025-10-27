
# # gemini_planner.py
# import google.generativeai as genai
# import json
# import os
# import requests

# # --- API KEY CONFIGURATION ---
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
# if not GOOGLE_API_KEY or not UNSPLASH_ACCESS_KEY:
#     raise ValueError("One or more API keys (GOOGLE_API_KEY, UNSPLASH_ACCESS_KEY) are not set!")
# genai.configure(api_key=GOOGLE_API_KEY)

# # --- IMAGE FETCHER FUNCTION ---
# def get_location_image_url(photo_query):
#     """Fetches a high-quality image URL from Unsplash based on a query."""
#     try:
#         url = f"https://api.unsplash.com/search/photos?query={photo_query}&per_page=1&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}"
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
#         if data['results']:
#             return data['results'][0]['urls']['regular']
#     except requests.exceptions.RequestException as e:
#         print(f"CRITICAL ERROR fetching image from Unsplash: {e}")
#     # A generic fallback image ensures the UI never looks broken
#     return "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1"

# # --- RECOMMENDATION FUNCTION ---
# def get_destination_recommendations(preferences):
#     """Gets recommendations based on detailed, multi-step user preferences."""
#     # Using Flash here is fine as it's a simpler, faster task.
#     model = genai.GenerativeModel('gemini-1.5-flash-latest')
#     prompt = f"""
#     You are an expert travel recommender. Based on the user's preferences ({preferences}), suggest 3 distinct destinations.
#     Your response MUST be a JSON object with a single key "destinations", which is a list of objects.
#     Each object must have these keys: "city", "country", "description", and "photo_query".
#     """
#     try:
#         response = model.generate_content(prompt)
#         json_string = response.text.replace('```json', '').replace('```', '').strip()
#         return json.loads(json_string)
#     except Exception as e:
#         print(f"CRITICAL ERROR in get_destination_recommendations: {e}")
#         return None

# # --- MASTER ITINERARY FUNCTION ---
# def generate_master_itinerary(destination, preferences):
#     """Generates a complete travel plan with ACCURATE budget calculations."""
#     # Using the Pro model here is better for complex reasoning and instruction following.
#     model = genai.GenerativeModel('gemini-1.5-pro-latest') 
#     prompt = f"""
#     You are a world-class travel consultant. Your primary goal is to generate a response that perfectly matches the requested JSON structure. Create a complete and highly detailed travel plan for a trip to {destination['city']}, {destination['country']}.

#     User Preferences: {preferences}

#     CRITICAL INSTRUCTIONS FOR BUDGETING: The "budget_breakdown" must be for a SINGLE PERSON.

#     The JSON structure MUST be exactly as follows:
#     {{"title": "...", "assumptions": {{"budget_per_person": "...", "origin": "...", "visa_status": "..."}}, "budget_breakdown": {{"flights": "...", "visa_fees": "...", "accommodation": "...", "intercity_travel": "...", "activities_and_food": "...", "contingency": "..."}}, "why_this_destination": "...", "detailed_itinerary": {{"Day 0": "...", "Day 1": {{"theme": "...", "morning": "...", "afternoon": "...", "evening": "..."}}}}, "booking_tips": {{"flights": "...", "trains": "...", "accommodation": "..."}}, "visa_and_insurance": "...", "checklist": [...], "photo_query": "..."}}
#     """
#     try:
#         response = model.generate_content(prompt)
#         json_string = response.text.replace('```json', '').replace('```', '').strip()
#         return json.loads(json_string)
#     except Exception as e:
#         print(f"CRITICAL ERROR in generate_master_itinerary: {e}")
#         return None

# # --- REVISION FUNCTION ---
# def revise_itinerary(original_plan, revision_request):
#     """Takes an existing itinerary and a user's request to revise it."""
#     model = genai.GenerativeModel('gemini-1.5-pro-latest')
#     prompt = f"""
#     You are a travel plan editor. You have already created the following travel plan: ORIGINAL PLAN: --- {original_plan} ---. The user has requested the following changes: USER'S REVISION REQUEST: --- "{revision_request}" ---. Your task is to generate a NEW, revised travel plan that incorporates the user's feedback. The output MUST be a single, clean JSON object with the exact same structure as the original plan (title, assumptions, detailed_itinerary, etc.). Modify the "detailed_itinerary" and any other relevant sections to reflect the user's request.
#     """
#     try:
#         response = model.generate_content(prompt)
#         json_string = response.text.replace('```json', '').replace('```', '').strip()
#         return json.loads(json_string)
#     except Exception as e:
#         print(f"CRITICAL ERROR in revise_itinerary: {e}")
#         return None
# gemini_planner.py
import google.generativeai as genai
import json
import os
import requests

# --- API KEY CONFIGURATION ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
if not GOOGLE_API_KEY or not UNSPLASH_ACCESS_KEY:
    raise ValueError("One or more API keys (GOOGLE_API_KEY, UNSPLASH_ACCESS_KEY) are not set!")
genai.configure(api_key=GOOGLE_API_KEY)

# --- IMAGE FETCHER FUNCTION ---
def get_location_image_url(photo_query):
    """Fetches a high-quality image URL from Unsplash based on a query."""
    try:
        url = f"https://api.unsplash.com/search/photos?query={photo_query}&per_page=1&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            return data['results'][0]['urls']['regular']
    except requests.exceptions.RequestException as e:
        print(f"CRITICAL ERROR fetching image from Unsplash: {e}")
    # A generic fallback image ensures the UI never looks broken
    return "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1"

# --- RECOMMENDATION FUNCTION ---
def get_destination_recommendations(preferences):
    """Gets recommendations based on detailed, multi-step user preferences."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    You are an expert travel recommender. Based on the user's preferences ({preferences}), suggest 3 distinct destinations.
    Your response MUST be a JSON object with a single key "destinations", which is a list of objects.
    Each object must have these keys: "city", "country", "description", and "photo_query".
    """
    try:
        response = model.generate_content(prompt)
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"CRITICAL ERROR in get_destination_recommendations: {e}")
        return None

# --- MASTER ITINERARY FUNCTION (UPDATED) ---
def generate_master_itinerary(destination, preferences):
    """Generates a complete travel plan with ACCURATE budget calculations and BOOKING LINKS."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    You are a world-class travel consultant. Your primary goal is to generate a response that perfectly matches the requested JSON structure.
    Create a complete and highly detailed travel plan for a trip to {destination['city']}, {destination['country']}.

    User Preferences: {preferences}

    CRITICAL INSTRUCTIONS:
    1.  **Budgeting**: The "budget_breakdown" must be for a SINGLE PERSON.
    2.  **Main Booking Links**: Generate real, functional search URLs for Google Flights and Booking.com based on the destination and origin.
    3.  **Detailed Itinerary**:
        * "Day 0" should be a simple string for travel/arrival.
        * All other days (Day 1, Day 2, etc.) MUST be objects.
        * For "morning", "afternoon", and "evening", you MUST provide an object with "activity" (a string description) and "booking_link" (a URL string).
        * If no specific booking is needed (e.g., "Explore the neighborhood"), the "booking_link" should be a relevant Google Maps search URL.
        * For specific activities (museums, tours, restaurants), provide a *direct info or booking link* (e.g., from GetYourGuide, Viator, or the official site).

    The JSON structure MUST be exactly as follows:
    {{
        "title": "Your Ultimate Trip to {destination['city']}",
        "assumptions": {{
            "budget_per_person": "...",
            "origin": "{preferences.get('departure_city', 'Not specified')}",
            "visa_status": "..."
        }},
        "budget_breakdown": {{
            "flights": "...",
            "visa_fees": "...",
            "accommodation": "...",
            "intercity_travel": "...",
            "activities_and_food": "...",
            "contingency": "..."
        }},
        "why_this_destination": "A paragraph on why this is a great choice.",
        "detailed_itinerary": {{
            "Day 0": "Travel from {preferences.get('departure_city', 'origin')} to {destination['city']}. Settle into your accommodation.",
            "Day 1": {{
                "theme": "Historical Immersion",
                "morning": {{"activity": "Visit the National Museum", "booking_link": "https://www.example.com/national-museum-tickets"}},
                "afternoon": {{"activity": "Guided tour of the Old City", "booking_link": "https://www.getyourguide.com/example-city/old-city-tour-t12345"}},
                "evening": {{"activity": "Dinner at 'Historic Tavern' (Reservation recommended)", "booking_link": "https://www.example.com/historic-tavern/reservations"}}
            }}
        }},
        "booking_tips": {{
            "flights": "...",
            "trains": "...",
            "accommodation": "..."
        }},
        "visa_and_insurance": "...",
        "checklist": ["Passport", "Visa (if required)", "..."],
        "photo_query": "{destination['city']} {destination['country']} landmarks",
        "main_booking_links": {{
            "flights": "https://www.google.com/flights?q=Flights+from+{preferences.get('departure_city', 'origin')}+to+{destination['city']}",
            "hotels": "https://www.booking.com/searchresults.html?ss={destination['city']},+{destination['country']}"
        }}
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"CRITICAL ERROR in generate_master_itinerary: {e}")
        return None

# --- REVISION FUNCTION (UPDATED) ---
def revise_itinerary(original_plan, revision_request):
    """Takes an existing itinerary and a user's request to revise it, maintaining the NEW structure."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    You are a travel plan editor. You have already created the following travel plan:
    ORIGINAL PLAN:
    ---
    {original_plan}
    ---

    The user has requested the following changes:
    USER'S REVISION REQUEST:
    ---
    "{revision_request}"
    ---

    Your task is to generate a NEW, revised travel plan that incorporates the user's feedback.
    The output MUST be a single, clean JSON object with the exact same structure as the original plan.
    This includes the "main_booking_links" and the detailed_itinerary structure where "morning", "afternoon", and "evening" are objects containing "activity" and "booking_link".
    """
    try:
        response = model.generate_content(prompt)
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"CRITICAL ERROR in revise_itinerary: {e}")
        return None
