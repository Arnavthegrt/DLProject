import requests

GEOAPIFY_API_KEY = '5af068e324024b6db743c4ee2efd87c1'

def verify_cities(city_list):
    """
    Takes a list of city names and verifies they are real using a simple geocode check.
    Returns a list of the verified city names.
    """
    if not city_list:
        return ["Gemini did not recommend any cities."]

    verified_cities = []
    for city in city_list:
        print(f"DEBUG: Verifying '{city}'...")
        # Use a simple API call to check if the place exists
        geocode_url = f"https://api.geoapify.com/v1/geocode/search?text={city}&format=json&limit=1&apiKey={GEOAPIFY_API_KEY}"
        try:
            response = requests.get(geocode_url).json()
            # If the API returns a result, we consider it verified
            if response.get('results'):
                verified_cities.append(city)
        except requests.exceptions.RequestException as e:
            print(f"API request error for {city}: {e}")
            continue # Skip to the next city on error

    return verified_cities