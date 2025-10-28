'''import google.generativeai as genai
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
'''






















'''
import google.generativeai as genai
import json
import os
import requests
import re # Import regex for pattern matching
import spacy # Import spaCy
import numpy as np

# --- Import Transformers for Summarization ---
try:
    from transformers import pipeline
    print("Transformers library loaded successfully.")
except ImportError:
    print("Transformers library not found. Please run: pip install transformers torch")
    pipeline = None

# --- Load spaCy Model ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model 'en_core_web_sm' loaded successfully.")
except IOError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None

# --- Load Summarization Model ---
summarizer = None
if pipeline:
    try:
        # Using a reliable summarization model
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        print("Summarization model loaded successfully.")
    except Exception as e:
        print(f"Could not load summarization model: {e}. Summarization will be disabled.")
else:
     print("Transformers not available. Summarization will be disabled.")


# --- API KEY CONFIGURATION ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
if not GOOGLE_API_KEY or not UNSPLASH_ACCESS_KEY:
    raise ValueError("One or more API keys (GOOGLE_API_KEY, UNSPLASH_ACCESS_KEY) are not set!")
genai.configure(api_key=GOOGLE_API_KEY)

# Use the stable -latest identifier for reliability
MODEL_FLASH = 'gemini-2.0-flash'
MODEL_PRO = 'gemini-2.0-flash' # Recommended for complex generation
LONG_TEXT_THRESHOLD = 35 # Words to trigger summarization

# --- IMAGE FETCHER FUNCTION ---
def get_location_image_url(photo_query):
    """Fetches a high-quality image URL from Unsplash based on a query."""
    print(f"[API Call]: Fetching image from Unsplash for query: {photo_query}")
    try:
        url = f"https://api.unsplash.com/search/photos?query={photo_query}&per_page=1&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get('results'):
            image_url = data['results'][0]['urls']['regular']
            print("[API Call]: Unsplash image found.")
            return image_url
    except requests.exceptions.RequestException as e:
        print(f"CRITICAL ERROR fetching image from Unsplash: {e}")
    print("[API Call]: Unsplash fetch failed, using fallback image.")
    return "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1"

# --- RECOMMENDATION FUNCTION ---
def get_destination_recommendations(preferences):
    """Gets recommendations based on detailed, multi-step user preferences."""
    print("[LLM Task]: Getting destination recommendations...")
    model = genai.GenerativeModel(MODEL_FLASH) # Flash is good here
    prompt = f"""
    You are an expert travel recommender. Based on the user's preferences ({preferences}), suggest 3 distinct destinations.
    Your response MUST be a JSON object with a single key "destinations", which is a list of objects.
    Each object must have these keys: "city", "country", "description", and "photo_query".
    """
    try:
        response = model.generate_content(prompt)
        # Check for safety blocks
        if not response.parts:
            print("CRITICAL ERROR in get_destination_recommendations: Response blocked.", response.prompt_feedback)
            return None
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"CRITICAL ERROR in get_destination_recommendations: {e}")
        return None

# --- MASTER ITINERARY FUNCTION ---
def generate_master_itinerary(destination, preferences):
    """Generates a complete travel plan including booking links."""
    print(f"[LLM Task]: Generating master itinerary for {destination.get('city')}...")
    # Using Pro model is recommended here for better structure following
    model = genai.GenerativeModel(MODEL_PRO)
    prompt = f"""
    You are a world-class travel consultant. Your primary goal is to generate a response that perfectly matches the requested JSON structure.
    Create a complete and highly detailed travel plan for a trip to {destination['city']}, {destination['country']}.

    User Preferences: {preferences}

    CRITICAL INSTRUCTIONS:
    1. Budgeting: The "budget_breakdown" must be for a SINGLE PERSON. It MUST include all keys: "flights", "visa_fees", "accommodation", "intercity_travel", "activities_and_food", "contingency". Provide estimates.
    2. Main Booking Links: Generate real, functional search URLs for Google Flights and Booking.com.
    3. Detailed Itinerary:
       * "Day 0" is a string.
       * All other days MUST be objects.
       * "morning", "afternoon", "evening" MUST provide an object with "activity" (string) and "booking_link" (URL string - use Google search if no direct link).

    The JSON structure MUST be exactly as follows:
    {{
      "title": "...", "assumptions": {{...}}, "budget_breakdown": {{...}}, "why_this_destination": "...",
      "detailed_itinerary": {{
        "Day 0": "...",
        "Day 1": {{ "theme": "...", "morning": {{"activity": "...", "booking_link": "..."}}, "afternoon": {{...}}, "evening": {{...}} }}
      }},
      "booking_tips": {{...}}, "visa_and_insurance": "...", "checklist": [...], "photo_query": "...",
      "main_booking_links": {{ "flights": "...", "hotels": "..." }}
    }}
    Ensure perfect JSON syntax, no trailing commas.
    """
    try:
        response = model.generate_content(prompt)
        if not response.parts:
            print("CRITICAL ERROR in generate_master_itinerary: Response blocked.", response.prompt_feedback)
            return None
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        # Basic validation
        parsed_json = json.loads(json_string)
        if "detailed_itinerary" not in parsed_json or "budget_breakdown" not in parsed_json:
             print("CRITICAL WARNING: Generated JSON missing core itinerary or budget keys.")
        return parsed_json
    except Exception as e:
        print(f"CRITICAL ERROR in generate_master_itinerary: {e}")
        return None

# --- LOCAL NLP-POWERED REVISION FUNCTIONS ---

def summarize_text(text):
    """Uses a local 'transformers' model to summarize long text."""
    print("[Local NLP Task]: Summarizing long text...")
    if summarizer is None:
        print("[Local NLP Error]: Summarizer model not loaded. Returning original text.")
        return text # Return original text if summarizer failed to load
    try:
        # Limit input text length if needed by the model
        max_input_length = summarizer.model.config.max_position_embeddings - 2 # Account for special tokens
        if len(text.split()) > max_input_length * 0.8: # Heuristic check
             print(f"[Local NLP Warning]: Input text might be too long for summarizer, truncating.")
             text = ' '.join(text.split()[:int(max_input_length * 0.8)])

        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        summary_text = summary[0]['summary_text']
        print(f"[Local NLP - Summary]: {summary_text}")
        return summary_text
    except Exception as e:
        print(f"CRITICAL ERROR in summarize_text: {e}")
        return text # Return original text on failure

BUDGET_KEYWORDS = ['cheap', 'cheaper', 'expensive', 'cost', 'costs', 'budget', 'price', 'pricey', 'affordable']
PACE_KEYWORDS = ['fast', 'slow', 'relax', 'relaxing', 'rush', 'rushing', 'boring', 'pace', 'leisurely']
ADVENTURE_THEMES = ['adventure', 'adventurous', 'exciting', 'hike', 'hiking', 'trekking', 'action', 'thrill']
RELAX_THEMES = ['relax', 'relaxing', 'beach', 'spa', 'chill', 'easy', 'leisure', 'rest']
NEGATIVE_THEMES = ['boring', 'dull', 'lame', 'bad', 'hate', 'terrible', 'dislike', 'remove']
POSITIVE_VERBS = ['add', 'include', 'want', 'prefer', 'like', 'love']

def analyze_revision_request_local_nlp(revision_request):
    """Uses spaCy for NER and keyword matching to analyze the user's request."""
    print(f"\n[Local NLP Task]: Analyzing short text with spaCy: '{revision_request}'")
    if nlp is None:
        print("SpaCy model not loaded. No NLP hint will be provided.")
        return "{}" # Return empty JSON string representation
    
    doc = nlp(revision_request)
    target_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "ORG", "FAC", "LOC", "EVENT", "PRODUCT"]]
    print(f"[Local NLP - NER]: Found entities: {target_entities}")

    # More robust day finding
    target_days_match = re.findall(r'(day\s*\d+|all\s*days?|entire\s*trip|whole\s*trip)', revision_request, re.IGNORECASE)
    if target_days_match:
         target_days = [d.replace(" ", "").title() if "day" in d.lower() else "all" for d in target_days_match]
    else:
         target_days = ["all"] # Default to all days if none specified
    target_days = list(set(target_days)) # Unique days

    lower_request = revision_request.lower()
    add_themes = list(set([word for word in ADVENTURE_THEMES + RELAX_THEMES if word in lower_request]))
    remove_themes = list(set([word for word in NEGATIVE_THEMES if word in lower_request]))

    # --- Rule-Based Intent Classification ---
    intent = "MODIFY_ACTIVITY" # Default intent
    if any(word in lower_request for word in BUDGET_KEYWORDS): intent = "REVISE_BUDGET"
    elif any(word in lower_request for word in PACE_KEYWORDS): intent = "REVISE_PACE"
    elif any(v in lower_request for v in POSITIVE_VERBS) and target_entities: intent = "ADD_ACTIVITY" # e.g., "add Eiffel Tower"
    elif target_entities and remove_themes: intent = "REPLACE_ACTIVITY" # e.g., "Louvre was boring" -> replace Louvre
    elif add_themes or remove_themes: intent = "REVISE_ACTIVITY_THEME" # e.g., "make it more relaxing"
    elif any(word in lower_request for word in NEGATIVE_THEMES): intent = "GENERAL_NEGATIVE" # Generic dislike

    analysis = {
        "intent": intent,
        "target_days": target_days,
        "target_entities": target_entities,
        "add_themes": add_themes,
        "remove_themes": remove_themes
    }
    
    analysis_json = json.dumps(analysis, indent=2)
    print(f"[Local NLP Result]: {analysis_json}")
    return analysis_json


# --- UNIFIED REVISION FUNCTION ---
def revise_itinerary(original_plan, revision_request):
    """
    Runs local NLP (Summarizer or NER) and passes hint to the LLM.
    """
    word_count = len(revision_request.split())
    nlp_hint = "{}" # Default empty hint

    if word_count > LONG_TEXT_THRESHOLD:
        print(f"[Router]: Request has {word_count} words (>{LONG_TEXT_THRESHOLD}). Using Summarizer.")
        nlp_hint = summarize_text(revision_request) if summarizer else revision_request # Fallback to full text
    else:
        print(f"[Router]: Request has {word_count} words (<={LONG_TEXT_THRESHOLD}). Using spaCy NER.")
        nlp_hint = analyze_revision_request_local_nlp(revision_request) if nlp else "{}" # Fallback to empty JSON

    print("[LLM Task]: Revising full itinerary with NLP hint...")
    # Use Pro model for revision consistency is generally better
    model = genai.GenerativeModel(MODEL_PRO) 
    
    original_plan_json = json.dumps(original_plan, indent=2) 

    prompt = f"""
    You are a travel plan editor. Original Plan: --- {original_plan_json} ---
    User Request: --- "{revision_request}" ---
    NLP Hint (analysis/summary of user request): --- {nlp_hint} ---
    
    TASK: Generate a NEW, revised travel plan incorporating the feedback, guided by the NLP Hint.
    If the hint indicates specific days or activities, focus changes there. If it's a summary, revise accordingly.
    Output MUST be a single, clean JSON object with the exact same structure as the original plan,
    including "main_booking_links" and the detailed_itinerary structure (with activity and booking_link objects).
    Ensure perfect JSON syntax. Update budget_breakdown if changes affect cost.
    """
    
    try:
        response = model.generate_content(prompt)
        if not response.parts:
            print("CRITICAL ERROR in revise_itinerary: Response blocked.", response.prompt_feedback)
            return original_plan # Return original on failure
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        # Add validation for revised plan structure if desired
        return json.loads(json_string)
    except Exception as e:
        print(f"CRITICAL ERROR in revise_itinerary: {e}")
        return original_plan # Return original plan on failure

# --- EXAMPLE USAGE (for testing this file directly) ---
if __name__ == "__main__":
    if nlp is None or summarizer is None:
        print("\n--- WARNING: A required local NLP model (spaCy or Transformers) is not loaded. ---")
        # Allow script to continue for testing core LLM functions
        
    user_preferences = {"duration": "5", "budget": "1000 EUR", "interests": ["History", "Food"], "pace": "Balanced", "departure_city": "New York"}
    example_destination = {"city": "Lisbon", "country": "Portugal"}
    
    print(f"\n--- Generating Master Itinerary for {example_destination['city']} ---")
    master_plan = generate_master_itinerary(example_destination, user_preferences)
    
    if not master_plan:
        print("Failed to generate master plan. Exiting test.")
        exit()
    
    master_plan['hero_image_url'] = get_location_image_url(master_plan.get('photo_query', 'travel'))
    print(f"\n--- Master Plan Generated (Title: {master_plan.get('title')}) ---")

    # Test short revision
    user_revision_1 = "day 1 National Museum sounds boring. Add a tuk-tuk tour instead."
    print(f"\n--- Handling Revision 1 (SHORT / NER Path) ---")
    revised_plan_1 = revise_itinerary(master_plan, user_revision_1)
    if revised_plan_1 and revised_plan_1 != master_plan:
        print("\n--- Revision 1 Complete! ---")
    else:
        print("\n--- Revision 1 Failed or No Change Made. ---")

    # Test long revision
    user_revision_2 = "Actually, the whole budget seems high. Find cheaper food options. Also, Day 3 feels too rushed, make it more relaxing perhaps with a beach visit."
    print(f"\n--- Handling Revision 2 (LONG / Summarizer Path) ---")
    current_plan_for_rev2 = revised_plan_1 or master_plan 
    revised_plan_2 = revise_itinerary(current_plan_for_rev2, user_revision_2)
    if revised_plan_2 and revised_plan_2 != current_plan_for_rev2:
        print("\n--- Revision 2 Complete! ---")
    else:
        print("\n--- Revision 2 Failed or No Change Made. ---")

'''
import google.generativeai as genai
import json
import os
import requests
import re
import spacy
import numpy as np
import pandas as pd # Add pandas for DataFrame creation
import joblib # Add joblib to load saved objects

# --- Import TensorFlow/Keras ---
try:
    import tensorflow as tf
    print("TensorFlow loaded successfully.")
except ImportError:
    print("TensorFlow not found. Please run: pip install tensorflow")
    tf = None

# --- (Local NLP imports for revision - spaCy, Transformers remain the same) ---
try: from transformers import pipeline; print("Transformers loaded.")
except ImportError: print("Transformers not found."); pipeline = None
try: nlp = spacy.load("en_core_web_sm"); print("SpaCy loaded.")
except IOError: print("SpaCy model not found."); nlp = None
summarizer = None
if pipeline:
    try: summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6"); print("Summarizer loaded.")
    except Exception as e: print(f"Could not load summarizer: {e}")
else: print("Summarization disabled.")

# --- API KEY CONFIGURATION ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
if not GOOGLE_API_KEY or not UNSPLASH_ACCESS_KEY:
    raise ValueError("Missing GOOGLE_API_KEY or UNSPLASH_ACCESS_KEY!")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Use gemini-1.5-flash-latest as requested ---
MODEL_FLASH = 'gemini-2.0-flash'
LONG_TEXT_THRESHOLD = 35

# --- Load Saved Preprocessing Objects ---
try:
    preprocessor = joblib.load('preprocessor.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    destination_classes = np.load('destination_classes.npy', allow_pickle=True)
    print("Loaded preprocessor, label encoder, and class names successfully.")
    NUM_DESTINATIONS = len(destination_classes) # Get the number of classes directly
    print(f"Model predicts {NUM_DESTINATIONS} possible destinations.")

except FileNotFoundError:
    print("ERROR: Could not find 'preprocessor.joblib', 'label_encoder.joblib', or 'destination_classes.npy'.")
    preprocessor = None; label_encoder = None; destination_classes = []; NUM_DESTINATIONS = 0
except Exception as e:
    print(f"Error loading preprocessing objects: {e}")
    preprocessor = None; label_encoder = None; destination_classes = []; NUM_DESTINATIONS = 0

# --- IMAGE FETCHER FUNCTION ---
def get_location_image_url(photo_query):
    # (Function remains the same)
    print(f"[API Call]: Fetching image from Unsplash for query: {photo_query}")
    try:
        url = f"https://api.unsplash.com/search/photos?query={photo_query}&per_page=1&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url); response.raise_for_status(); data = response.json()
        if data.get('results'): print("[API Call]: Unsplash image found."); return data['results'][0]['urls']['regular']
    except requests.exceptions.RequestException as e: print(f"CRITICAL ERROR fetching image from Unsplash: {e}")
    print("[API Call]: Unsplash fetch failed, using fallback image.")
    return "https://images.unsplash.com/photo-1476514525535-07fb_b4ae5f1"

# --- HELPER FUNCTION TO ENRICH DESTINATIONS USING GEMINI ---
def enrich_destinations_with_gemini(destination_names, preferences):
    """Takes a list of city/country names and uses Gemini to add descriptions and photo queries."""
    print("[LLM Task]: Enriching predicted destinations with descriptions...")
    model = genai.GenerativeModel(MODEL_FLASH)
    
    # Create a simple list of destinations for the prompt
    dest_list_str = "\n".join([f"- {name}" for name in destination_names])
    
    prompt = f"""
    Given the following user preferences: {preferences}
    
    And the following list of potential destinations predicted by a model:
    {dest_list_str}

    Your task is to generate descriptions and photo queries for these destinations.
    Return a JSON object where the keys are the original destination names (e.g., "Paris, France")
    and the values are objects containing:
    - "description": A compelling, one-paragraph description highlighting why this destination fits the user preferences.
    - "photo_query": A simple, effective search query for Unsplash (e.g., "Eiffel Tower Paris night").

    Example for one destination:
    "Paris, France": {{
        "description": "Paris offers iconic history...",
        "photo_query": "Eiffel Tower Paris night"
    }}
    
    Output ONLY the JSON object.
    """
    try:
        response = model.generate_content(prompt)
        if not response.parts:
            print("CRITICAL ERROR in enrich_destinations_with_gemini: Response blocked.", response.prompt_feedback)
            return {} # Return empty dict on failure
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        enriched_data = json.loads(json_string)
        print("[LLM Task]: Enrichment successful.")
        return enriched_data
    except Exception as e:
        print(f"CRITICAL ERROR in enrich_destinations_with_gemini: {e}")
        return {} # Return empty dict on failure


# --- RECOMMENDATION FUNCTION (Using Local H5 Model + Gemini Enrichment) ---
def get_destination_recommendations(preferences):
    """Gets recommendations using the local .h5 model, then enriches with Gemini."""
    print("[Local Model Task]: Getting recommendations from destination_predictor_model.h5...")

    if tf is None or preprocessor is None or label_encoder is None or NUM_DESTINATIONS == 0:
        print("ERROR: TensorFlow, preprocessor objects, or destination classes not available.")
        # Fallback to pure Gemini recommendation if local model fails
        print("[Fallback]: Using Gemini API for recommendations instead.")
        return get_destination_recommendations_gemini_fallback(preferences) # Need a fallback function

    # --- 1. Load the Keras Model ---
    model_path = 'destination_predictor_model.h5'
    try:
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {os.path.abspath(model_path)}")
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded local model from {model_path}")
    except Exception as e:
        print(f"CRITICAL ERROR loading local model '{model_path}': {e}")
        return get_destination_recommendations_gemini_fallback(preferences) # Fallback

    # --- 2. Convert Preferences to DataFrame ---
    try:
        # (Preprocessing code remains the same as your last working version)
        print("Creating DataFrame from preferences...")
        input_data_raw = {}
        input_data_raw['duration_days'] = pd.to_numeric(preferences.get('duration'), errors='coerce')
        input_data_raw['num_people'] = pd.to_numeric(preferences.get('people'), errors='coerce')
        budget_str = preferences.get('budget', '0 INR'); budget_num = 0.0
        try: match = re.search(r'(\d+(\.\d+)?)', budget_str); budget_num = float(match.group(1)) if match else 0.0
        except Exception: print(f"Warning: Could not parse budget '{budget_str}'. Using 0.")
        input_data_raw['budget_per_person_inr'] = budget_num
        start_month = 1; start_date_str = preferences.get('start_date')
        if start_date_str:
             try: input_data_raw['start_date'] = pd.to_datetime(start_date_str); start_month = input_data_raw['start_date'].month
             except Exception: print(f"Warning: Could not parse start_date '{start_date_str}'. Using default month 1.")
        input_data_raw['start_month'] = start_month
        input_data_raw['departure_city'] = preferences.get('departure_city', 'Unknown')
        input_data_raw['preferred_pace'] = preferences.get('pace', 'Balanced')
        input_data_raw['accommodation_style'] = preferences.get('accommodation_style', 'Mid-range Hotels')
        interests_list = preferences.get('interests', []); input_data_raw['interests'] = '|'.join(interests_list) if isinstance(interests_list, list) else ''
        input_df = pd.DataFrame([input_data_raw])
        known_interest_cols_base = ['Adventure', 'Beaches', 'Food', 'History', 'Nature', 'Nightlife']
        known_interest_cols_prefixed = [f'interest_{col}' for col in known_interest_cols_base]
        if 'interests' in input_df.columns:
             interest_dummies = input_df['interests'].str.get_dummies(sep='|'); interest_dummies.columns = [f'interest_{col}' for col in interest_dummies.columns]
             for col in known_interest_cols_prefixed:
                  if col not in interest_dummies.columns: interest_dummies[col] = 0
             interest_dummies = interest_dummies[known_interest_cols_prefixed]
             input_df = pd.concat([input_df.drop('interests', axis=1), interest_dummies], axis=1)
        else:
              for col in known_interest_cols_prefixed: input_df[col] = 0
        if 'start_date' in input_df.columns: input_df = input_df.drop('start_date', axis=1)
        try:
             expected_cols = list(preprocessor.feature_names_in_)
             for col in expected_cols:
                  if col not in input_df.columns:
                       input_df[col] = 0 if col in ['duration_days', 'num_people', 'budget_per_person_inr', 'start_month'] or col.startswith('interest_') else 'Unknown'
             input_df = input_df[expected_cols]
        except AttributeError: print("Warning: Could not get feature names from preprocessor.")
        print("--- DataFrame before preprocessing ---"); input_df.info(); print(input_df.head())
    except Exception as e:
         print(f"CRITICAL ERROR creating DataFrame for local model: {e}")
         return get_destination_recommendations_gemini_fallback(preferences) # Fallback

    # --- 3. Apply Preprocessor ---
    try:
        print("Applying loaded preprocessor..."); input_processed = preprocessor.transform(input_df)
        if hasattr(input_processed, "toarray"): print("Converting sparse matrix..."); input_processed = input_processed.toarray()
        print(f"Processed input data shape: {input_processed.shape}"); print(f"Processed input data type: {input_processed.dtype}")
        expected_feature_count = model.input_shape[1]
        if input_processed.shape[1] != expected_feature_count: raise ValueError(f"Preprocessor output count ({input_processed.shape[1]}) != model expected count ({expected_feature_count})")
    except Exception as e:
        print(f"CRITICAL ERROR applying preprocessor: {e}"); print("--- DataFrame dtypes just before error ---"); input_df.info()
        return get_destination_recommendations_gemini_fallback(preferences) # Fallback

    # --- 4. Make Prediction ---
    try:
        print("Making prediction..."); input_processed = np.asarray(input_processed).astype(np.float32)
        predictions = model.predict(input_processed); print(f"Raw prediction shape: {predictions.shape}")
    except Exception as e:
        print(f"CRITICAL ERROR during prediction: {e}")
        return get_destination_recommendations_gemini_fallback(preferences) # Fallback

    # --- 5. Postprocess Output (Get Top Names) ---
    try:
        print("Postprocessing prediction...")
        if len(predictions[0]) != NUM_DESTINATIONS: print(f"Warning: Model output size ({len(predictions[0])}) != expected ({NUM_DESTINATIONS}).")
        
        top_indices = np.argsort(predictions[0])[::-1][:3]
        predicted_destination_names = label_encoder.inverse_transform(top_indices)
        print(f"Top 3 predicted destination names: {predicted_destination_names}")

        if not predicted_destination_names.size: # Check if empty
             print("Warning: No valid destinations predicted by local model.")
             return get_destination_recommendations_gemini_fallback(preferences) # Fallback

    except Exception as e:
        print(f"CRITICAL ERROR postprocessing prediction: {e}")
        return get_destination_recommendations_gemini_fallback(preferences) # Fallback

    # --- 6. Enrich with Gemini ---
    enriched_data = enrich_destinations_with_gemini(predicted_destination_names, preferences)
    
    # --- 7. Format Final Output ---
    final_recommendations = []
    for dest_name in predicted_destination_names:
        city_country = dest_name.split(',')
        city = city_country[0].strip()
        country = city_country[-1].strip() if len(city_country) > 1 else "Unknown"
        
        # Get enriched data if available
        enrichment = enriched_data.get(dest_name, {})
        description = enrichment.get("description", f"Explore the wonders of {dest_name}.") # Default description
        photo_query = enrichment.get("photo_query", f"{city} {country} travel") # Default query
        
        final_recommendations.append({
            "city": city,
            "country": country,
            "description": description,
            "photo_query": photo_query
        })
        
    result = {"destinations": final_recommendations}
    print(f"Final recommendations after enrichment: {result}")
    return result

# --- FALLBACK GEMINI RECOMMENDATION FUNCTION ---
def get_destination_recommendations_gemini_fallback(preferences):
    """Fallback function to get recommendations ONLY using Gemini API."""
    print("[Fallback LLM Task]: Getting recommendations via Gemini API...")
    model = genai.GenerativeModel(MODEL_FLASH)
    prompt = f"""
    You are an expert travel recommender. Based ONLY on the user's preferences ({preferences}), suggest 3 distinct destinations suitable for them.
    Your response MUST be a JSON object with a single key "destinations", which is a list of objects.
    Each object must have these keys: "city", "country", "description", and "photo_query".
    """
    try:
        response = model.generate_content(prompt)
        if not response.parts:
            print("CRITICAL ERROR in Fallback Gemini Rec: Response blocked.", response.prompt_feedback)
            return None
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"CRITICAL ERROR in Fallback Gemini Rec: {e}")
        return None


# --- MASTER ITINERARY FUNCTION ---
def generate_master_itinerary(destination, preferences):
    """Generates a complete travel plan including booking links, using Gemini."""
    print(f"[LLM Task]: Generating master itinerary for {destination.get('city')}...")
    model = genai.GenerativeModel(MODEL_FLASH) # Using Flash as requested
    prompt = f"""
    You are a world-class travel consultant. Your goal is to generate a response that perfectly matches the requested JSON structure.
    Create a complete and highly detailed travel plan for a trip to {destination['city']}, {destination['country']}.
    User Preferences: {preferences}
    CRITICAL INSTRUCTIONS:
    1. Budgeting: "budget_breakdown" must be for a SINGLE PERSON and include ALL keys: "flights", "visa_fees", "accommodation", "intercity_travel", "activities_and_food", "contingency".
    2. Main Booking Links: Generate real, functional search URLs for Google Flights and Booking.com in "main_booking_links".
    3. Detailed Itinerary: "Day 0" is a string. Other days MUST be objects. "morning", "afternoon", "evening" MUST be objects with "activity" (string) and "booking_link" (URL string - use Google search if no direct link).
    The JSON structure MUST be exactly:
    {{"title": "...", "assumptions": {{...}}, "budget_breakdown": {{...}}, "why_this_destination": "...", "detailed_itinerary": {{...}}, "booking_tips": {{...}}, "visa_and_insurance": "...", "checklist": [...], "photo_query": "...", "main_booking_links": {{...}} }}
    Ensure perfect JSON syntax.
    """
    try:
        response = model.generate_content(prompt)
        if not response.parts:
            print("CRITICAL ERROR in generate_master_itinerary: Response blocked.", response.prompt_feedback)
            return None
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        parsed_json = json.loads(json_string)
        if "detailed_itinerary" not in parsed_json or "budget_breakdown" not in parsed_json:
             print("CRITICAL WARNING: Generated JSON missing core itinerary or budget keys.")
        return parsed_json
    except Exception as e:
        print(f"CRITICAL ERROR in generate_master_itinerary: {e}")
        return None

# --- LOCAL NLP-POWERED REVISION FUNCTIONS ---
# (summarize_text, analyze_revision_request_local_nlp remain the same)
def summarize_text(text):
    print("[Local NLP Task]: Summarizing long text...")
    if summarizer is None: return text
    try:
        # Simplified truncation
        max_len = 500
        if len(text) > max_len: text = text[:max_len]
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e: print(f"ERROR in summarize_text: {e}"); return text

BUDGET_KEYWORDS=['cheap','expensive','cost','budget','price'];PACE_KEYWORDS=['fast','slow','relax','pace'];ADVENTURE_THEMES=['adventure','hike','action'];RELAX_THEMES=['relax','beach','spa','chill'];NEGATIVE_THEMES=['boring','bad','hate','remove'];POSITIVE_VERBS=['add','include','want']
def analyze_revision_request_local_nlp(revision_request):
    print(f"\n[Local NLP Task]: Analyzing short text: '{revision_request}'")
    if nlp is None: return "{}"
    doc=nlp(revision_request); target_entities=[ent.text for ent in doc.ents if ent.label_ in ["GPE","ORG","FAC","LOC","EVENT"]]; target_days=[d.title() for d in re.findall(r'day\s*\d+',revision_request,re.I)] or ["all"]
    lower_req=revision_request.lower(); add_themes=list(set([w for w in ADVENTURE_THEMES+RELAX_THEMES if w in lower_req])); remove_themes=list(set([w for w in NEGATIVE_THEMES if w in lower_req]))
    intent="MODIFY";
    if any(w in lower_req for w in BUDGET_KEYWORDS): intent="BUDGET"
    elif any(w in lower_req for w in PACE_KEYWORDS): intent="PACE"
    elif any(v in lower_req for v in POSITIVE_VERBS) and target_entities: intent="ADD"
    elif target_entities and remove_themes: intent="REPLACE"
    elif add_themes or remove_themes: intent="THEME"
    analysis={"intent":intent,"days":target_days,"entities":target_entities,"add":add_themes,"remove":remove_themes}
    analysis_json=json.dumps(analysis,indent=2); print(f"[Local NLP Result]: {analysis_json}"); return analysis_json

# --- UNIFIED REVISION FUNCTION ---
def revise_itinerary(original_plan, revision_request):
    """Runs local NLP and passes hint to the LLM."""
    word_count = len(revision_request.split()); nlp_hint = "{}"
    if word_count > LONG_TEXT_THRESHOLD: nlp_hint = summarize_text(revision_request) if summarizer else revision_request
    else: nlp_hint = analyze_revision_request_local_nlp(revision_request) if nlp else "{}"
    print("[LLM Task]: Revising itinerary with NLP hint..."); model = genai.GenerativeModel(MODEL_FLASH) # Use Flash for revision
    original_plan_json = json.dumps(original_plan, indent=2)
    prompt = f"""Original Plan: --- {original_plan_json} ---
User Request: --- "{revision_request}" ---
NLP Hint: --- {nlp_hint} ---
TASK: Generate a NEW, revised travel plan incorporating feedback, guided by the NLP Hint. Output MUST be a single, clean JSON object with the exact same structure as the original (including main_booking_links, activity/booking_link objects). Ensure perfect JSON."""
    try:
        response = model.generate_content(prompt)
        if not response.parts: print("CRITICAL ERROR in revise_itinerary: Blocked.", response.prompt_feedback); return original_plan
        return json.loads(response.text.replace('```json', '').replace('```', '').strip())
    except Exception as e: print(f"CRITICAL ERROR in revise_itinerary: {e}"); return original_plan

# --- EXAMPLE USAGE (for testing this file directly) ---
if __name__ == "__main__":
    if nlp is None or summarizer is None: print("\n--- WARNING: Local NLP model not loaded. ---")
    if preprocessor is None: print("--- WARNING: Preprocessor not loaded. Local recommendations disabled. ---")

    test_preferences = {"duration": "5", "budget": "1000 EUR", "interests": ["History", "Food"], "pace": "Balanced", "departure_city": "New York", "start_date": "2026-03-15", "people": "2", "accommodation_style": "Mid-range"}

    print("\n--- Testing Local H5 Recommendation ---")
    recommendations_h5 = get_destination_recommendations(test_preferences)

    if recommendations_h5 and recommendations_h5.get("destinations"):
        print("\n--- Local H5 Recs + Enrichment Successful ---")
        top_dest = recommendations_h5['destinations'][0]
        itinerary_prefs = test_preferences.copy(); itinerary_prefs['transportation'] = 'Public Transport'
        print(f"\n--- Generating Master Itinerary for {top_dest['city']} ---")
        master_plan = generate_master_itinerary(top_dest, itinerary_prefs)
        if master_plan: print(f"\n--- Master Plan Generated (Title: {master_plan.get('title')}) ---")
        else: print("\n--- Failed: Master Itinerary Generation ---")
    else:
        print("\n--- Failed: Local H5 Recommendations or Enrichment ---")

