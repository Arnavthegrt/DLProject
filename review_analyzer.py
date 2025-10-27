import google.generativeai as genai
import json
import os

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=GOOGLE_API_KEY)

def scrape_reviews_placeholder(topic):
    """
    A placeholder function that simulates web scraping.
    Returns a pre-saved list of reviews for a given topic.
    """
    print(f"DEBUG: 'Scraping' placeholder reviews for: {topic}...")
    # In a real-world app, you'd use BeautifulSoup/Scrapy here.
    # For a project demo, this is far more reliable.
    return [
        f"I visited {topic} and it was amazing! The food was incredible and the historical sites were breathtaking. A must-see.",
        f"A beautiful destination, but it was far too crowded in July. Getting a table at a good restaurant was a nightmare. Plan to go off-season.",
        f"Mixed feelings about {topic}. The beaches were nice and clean, but the city felt a bit overpriced. Good for a short visit.",
        f"The local culture in {topic} is rich and vibrant. We spent days just exploring the old town. Highly recommend for history buffs.",
        f"Absolutely stunning scenery. However, public transport was a bit lacking, so I'd recommend renting a car to explore properly."
    ]

def analyze_reviews_with_gemini(reviews):
    """
    Uses Gemini to perform sentiment analysis and summarization on a list of reviews.
    """
    if not reviews:
        return {"error": "No reviews to analyze."}

    all_reviews_text = "\n".join(reviews)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
    You are a travel review analyst. Analyze the following user reviews and return a single, clean JSON object with the following keys:
    - "overall_sentiment": A single word: "Positive", "Neutral", or "Negative".
    - "positive_points": A bulleted list of key positive aspects mentioned (e.g., ["Good Food", "Beautiful Scenery"]).
    - "negative_points": A bulleted list of key negative aspects mentioned (e.g., ["Crowded", "Expensive"]).
    - "summary": A concise, one-paragraph summary for a prospective traveler.

    Reviews to analyze:
    ---
    {all_reviews_text}
    ---
    """
    
    try:
        response = model.generate_content(prompt)
        json_string = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"Error processing Gemini review analysis: {e}")
        return None