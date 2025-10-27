
# # app.py
# from flask import Flask, render_template, request
# import json
# from gemini_planner import get_destination_recommendations, generate_master_itinerary, get_location_image_url, revise_itinerary

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/details', methods=['POST'])
# def details():
#     preferences = {
#         'departure_city': request.form['departure_city'],
#         'duration': request.form['duration'],
#         'people': request.form['people'],
#         'budget': f"{request.form['budget_amount']} {request.form['currency']}",
#     }
#     return render_template('details.html', preferences=preferences)

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     preferences = {
#         'departure_city': request.form['departure_city'],
#         'duration': request.form['duration'],
#         'people': request.form['people'],
#         'budget': request.form['budget'],
#         'pace': request.form['pace'],
#         'accommodation_style': request.form['accommodation_style'],
#         'interests': request.form.getlist('interests')
#     }
#     recommendations = get_destination_recommendations(preferences)
#     if not recommendations or 'destinations' not in recommendations:
#         return "Sorry, there was an error getting recommendations."
#     return render_template('recommendations.html', recommendations=recommendations, preferences=preferences)

# @app.route('/itinerary', methods=['POST'])
# def itinerary():
#     chosen_destination_str = request.form['chosen_destination']
#     city, country = chosen_destination_str.split('|')
#     destination = {'city': city, 'country': country}
#     preferences = {
#         'departure_city': request.form['departure_city'],
#         'duration': request.form['duration'],
#         'people': request.form['people'],
#         'budget': request.form['budget'],
#         'pace': request.form.get('pace'),
#         'accommodation_style': request.form.get('accommodation_style'),
#         'interests': request.form['interests'].split(','),
#         'transportation': request.form['transportation']
#     }
#     plan = generate_master_itinerary(destination, preferences)
#     if not plan:
#         return "Sorry, there was an error generating the itinerary."
    
#     photo_query = plan.get('photo_query', f"{city} {country}")
#     background_image_url = get_location_image_url(photo_query)
#     return render_template('itinerary.html', plan=plan, background_image_url=background_image_url)

# # --- NEW REVISION ROUTE ---
# @app.route('/revise', methods=['POST'])
# def revise():
#     """Handles requests to modify an existing itinerary."""
#     original_plan_str = request.form['original_plan']
#     revision_request = request.form['revision_request']
    
#     revised_plan = revise_itinerary(original_plan_str, revision_request)
#     if not revised_plan:
#         return "Sorry, there was an error revising the itinerary."

#     # Re-fetch the background image for the revised plan
#     photo_query = revised_plan.get('photo_query', "travel")
#     background_image_url = get_location_image_url(photo_query)

#     return render_template('itinerary.html', plan=revised_plan, background_image_url=background_image_url)

# if __name__ == '__main__':
#     app.run(debug=True)

# app.py
# app.py
from flask import Flask, render_template, request
import json
from gemini_planner import get_destination_recommendations, generate_master_itinerary, get_location_image_url, revise_itinerary

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details', methods=['POST'])
def details():
    preferences = {
        'departure_city': request.form['departure_city'],
        'duration': request.form['duration'],
        'people': request.form['people'],
        'budget': f"{request.form['budget_amount']} {request.form['currency']}",
        'start_date': request.form.get('start_date') 
    }
    return render_template('details.html', preferences=preferences)

@app.route('/recommend', methods=['POST'])
def recommend():
    preferences = {
        'departure_city': request.form['departure_city'],
        'duration': request.form['duration'],
        'people': request.form['people'],
        'budget': request.form['budget'],
        'pace': request.form['pace'],
        'accommodation_style': request.form['accommodation_style'],
        'interests': request.form.getlist('interests'),
        'start_date': request.form.get('start_date') 
    }
    recommendations = get_destination_recommendations(preferences)
    if not recommendations or 'destinations' not in recommendations:
        return "Sorry, there was an error getting recommendations."
    return render_template('recommendations.html', recommendations=recommendations, preferences=preferences)

@app.route('/itinerary', methods=['POST'])
def itinerary():
    chosen_destination_str = request.form['chosen_destination']
    city, country = chosen_destination_str.split('|')
    destination = {'city': city, 'country': country}
    
    preferences = {
        'departure_city': request.form['departure_city'],
        'duration': request.form['duration'],
        'people': request.form['people'],
        'budget': request.form['budget'],
        'pace': request.form.get('pace'),
        'accommodation_style': request.form.get('accommodation_style'),
        'interests': request.form['interests'].split(','),
        'transportation': request.form['transportation'],
        'start_date': request.form.get('start_date', 'Not specified') 
    }

    plan = generate_master_itinerary(destination, preferences)
    if not plan:
        return "Sorry, there was an error generating the itinerary."
    
    photo_query = plan.get('photo_query', f"{city} {country}")
    background_image_url = get_location_image_url(photo_query)
    return render_template('itinerary.html', plan=plan, background_image_url=background_image_url)

@app.route('/revise', methods=['POST'])
def revise():
    """Handles requests to modify an existing itinerary."""
    original_plan_str = request.form['original_plan']
    revision_request = request.form['revision_request']
    
    revised_plan = revise_itinerary(original_plan_str, revision_request)
    if not revised_plan:
        return "Sorry, there was an error revising the itinerary."

    photo_query = revised_plan.get('photo_query', "travel")
    background_image_url = get_location_image_url(photo_query)

    return render_template('itinerary.html', plan=revised_plan, background_image_url=background_image_url)

# --- THIS IS THE MISSING ROUTE ---
@app.route('/bookings', methods=['POST'])
def bookings():
    """Displays the booking hub page."""
    try:
        # Get the full plan JSON string from the hidden form field
        plan_str = request.form['original_plan']
        # Convert it back into a Python dictionary
        plan = json.loads(plan_str)
        
        # Extract the main links to pass to the template
        main_links = plan.get('main_booking_links', {})
        
        # Render the new bookings.html template
        return render_template('bookings.html', plan=plan, main_links=main_links)
    except Exception as e:
        print(f"CRITICAL ERROR in /bookings route: {e}")
        return "Sorry, there was an error loading the booking page."

if __name__ == '__main__':
    app.run(debug=True)

