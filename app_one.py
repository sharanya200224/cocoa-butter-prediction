from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the pre-trained ML model from .pkl file
try:
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully at {datetime.now().strftime('%I:%M %p IST, %A, %B %d, %Y')}")
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Please ensure xgboost_model.pkl is in the same directory as app.py.")
    model = None
except Exception as e:
    print(f"Error loading model with pickle: {e}. Please check the model file compatibility.")
    model = None

if model is None:
    raise Exception("Model loading failed. Application cannot proceed without a valid model.")

# Define feature names matching the HTML input fields
feature_names = [
    'oleic_acid', 'stearic_acid', 'palmitic_acid', 'melting_point', 'solid_fat',
    'viscosity', 'moisture', 'sos', 'pop', 'pos', 'lead', 'cadmium',
    'refining_time', 'conching_time', 'temperature', 'nir', 'ftir'
]

# Function to categorize quality based on score (scaled to 0-100)
def categorize_quality(score):
    if score >= 80:
        return 'good'
    elif score >= 50:
        return 'medium'
    else:
        return 'low'

# Function to determine skin type based on oleic_acid
def determine_skin_type(oleic_acid):
    if 26.09 <= oleic_acid <= 33.0:
        return 'normal', 'Well-balanced hydration suitable for normal skin.'
    elif oleic_acid > 33.0:
        return 'oily', 'Higher oleic acid may suit oily skin with proper refining.'
    elif 20.0 <= oleic_acid < 26.09:
        return 'dry', 'Lower oleic_acid helps hydrate dry skin.'
    else:
        return 'combination', 'Mixed properties suitable for combination skin.'

# Function to calculate compound match percentage
def calculate_compound_match(skin_type, oleic_acid, sos, pop_pos):
    base_match = 50  # Base match percentage
    if skin_type == 'normal' and 26.09 <= oleic_acid <= 33.0:
        base_match += 30
    elif skin_type == 'oily' and oleic_acid > 33.0:
        base_match += 30
    elif skin_type == 'dry' and 20.0 <= oleic_acid < 26.09:
        base_match += 30
    elif skin_type == 'combination':
        base_match += 20
    # Adjust based on triglycerides and POP/POS
    if sos >= 30:
        base_match += 10
    if 20 <= pop_pos <= 25:
        base_match += 10
    return min(max(base_match, 0), 100)

# Function to determine recommendation level
def determine_recommendation_level(quality_score, compound_match):
    if quality_score >= 80 and compound_match >= 80:
        return 'High'
    elif quality_score >= 50 and compound_match >= 50:
        return 'Medium'
    else:
        return 'Low'

# Function to generate skin concerns
def generate_skin_concerns(skin_type):
    concerns = {
        'dry': 'Texture, Hydration',
        'oily': 'Oil Control, Pore Clogging',
        'normal': 'Maintain Balance',
        'combination': 'Zonal Hydration, Oil Control',
        'sensitive': 'Irritation, Barrier Protection'
    }
    return concerns.get(skin_type, 'General Skin Care')

# Function to generate recommendation note
def generate_recommendation_note(quality_score, skin_type):
    if quality_score < 50:
        return 'Consider enhanced triglyceride formula for better moisture retention.'
    elif skin_type in ['oily', 'combination']:
        return 'Use lighter formulations and refine to avoid excess oil.'
    else:
        return 'Maintain current composition for optimal use.'

# Function to generate recommended products
def generate_recommended_products(skin_type, quality_category):
    products_by_type = {
        'dry': [
            {'name': 'Hydrating Body Butter', 'desc': 'Rich formula with high SOS and Oleic acid for deep moisturization', 'match': 92, 'icon': 'fa-pump-soap', 'compounds': ['High SOS', 'Oleic Acid', 'Vitamin E']},
            {'name': 'Overnight Repair Cream', 'desc': 'Intensive repair with high Stearic Acid and POS for skin barrier', 'match': 87, 'icon': 'fa-moon', 'compounds': ['High Stearic', 'POS', 'Ceramides']}
        ],
        'oily': [
            {'name': 'Matte Finish Lotion', 'desc': 'Light formula with POP/POS for oil control', 'match': 85, 'icon': 'fa-spray-can', 'compounds': ['POP/POS', 'Salicylic Acid', 'Mattifiers']},
            {'name': 'Oil-Free Balm', 'desc': 'Non-comedogenic with low Oleic Acid', 'match': 80, 'icon': 'fa-leaf', 'compounds': ['Low Oleic', 'Aloe Vera', 'Zinc Oxide']}
        ],
        'normal': [
            {'name': 'Daily Moisturizer', 'desc': 'Balanced Oleic and Stearic Acid for everyday use', 'match': 90, 'icon': 'fa-bath', 'compounds': ['Oleic Acid', 'Stearic Acid', 'Antioxidants']},
            {'name': 'Light Cream', 'desc': 'Gentle hydration with POP/POS', 'match': 88, 'icon': 'fa-spa', 'compounds': ['POP/POS', 'Hyaluronic Acid', 'Vitamin C']}
        ],
        'combination': [
            {'name': 'Zonal Hydration Gel', 'desc': 'Targets dry areas with SOS, controls oil with POP/POS', 'match': 83, 'icon': 'fa-adjust', 'compounds': ['SOS', 'POP/POS', 'Mattifiers']},
            {'name': 'Dual-Action Balm', 'desc': 'Hydrates dry zones, mattifies oily zones', 'match': 79, 'icon': 'fa-balance-scale', 'compounds': ['Oleic Acid', 'Stearic Acid', 'Zinc']}
        ],
        'sensitive': [
            {'name': 'Calming Balm', 'desc': 'Gentle formula for sensitive skin', 'match': 85, 'icon': 'fa-heart', 'compounds': ['Chamomile', 'Oat Extract', 'Vitamin E']},
            {'name': 'Barrier Repair Cream', 'desc': 'Strengthens skin barrier', 'match': 82, 'icon': 'fa-shield-alt', 'compounds': ['Ceramides', 'Niacinamide', 'Shea Butter']}
        ]
    }
    return products_by_type.get(skin_type, products_by_type['normal'])

# Function to describe how cocoa butter works
def how_cocoa_butter_works(skin_type, oleic_acid, sos):
    fatty_acids = f"Stearic and palmitic acids form a protective barrier, enhancing {skin_type} skin resilience with {oleic_acid:.1f}% Oleic Acid."
    triglycerides = f"SOS triglycerides at {sos:.1f}% create a moisture-locking matrix, ideal for {skin_type} skin hydration."
    antioxidants = "Natural polyphenols neutralize free radicals, reducing aging signs across all skin types."
    return {'fatty_acids': fatty_acids, 'triglycerides': triglycerides, 'antioxidants': antioxidants}

# Define refining steps and products based on quality category
refining_steps = {
    'good': ['No additional refining needed'],
    'medium': ['Filtration with activated carbon', 'Deodorization with steam', 'Winterization'],
    'low': ['Filtration with activated carbon', 'Deodorization with steam', 'Fractionation', 'Winterization']
}

products = {
    'good': [{'name': 'Lip Balm', 'desc': 'High-quality moisturization'}, {'name': 'Premium Soap', 'desc': 'Luxury cleansing'}],
    'medium': [{'name': 'Body Lotion', 'desc': 'Moisturizing with medium absorption'}, {'name': 'Bar Soap', 'desc': 'Moisturizing cleansing'}],
    'low': [{'name': 'Industrial Lubricant', 'desc': 'Non-cosmetic use'}, {'name': 'Candle Wax', 'desc': 'Basic coating'}]
}

# Route for home page (home_page.html)
@app.route('/')
def home():
    return render_template('home_page.html')

# Route for skin type recommendation page (skin.html)
@app.route('/skin')
def skin():
    return render_template('skin_type1.html')

# Route for butter extraction process page (processing.html)
@app.route('/processing')
def processing():
    return render_template('skin.html')

# Route for additional page (index1.html)
@app.route('/index1')
def index1():
    return render_template('index1.html')

# Route for quality analysis using the ML model
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        print(f"Received data: {data}")  # Log received data for debugging

        # Extract and validate input data
        input_data = [float(data.get(feature, 0)) for feature in feature_names]
        print(f"Input data array: {input_data}")  # Log the array sent to the model
        if any(v is None or not isinstance(v, (int, float)) for v in input_data):
            return jsonify({'error': 'Invalid input data'}), 400

        # Convert to numpy array and reshape for prediction
        input_data = np.array(input_data).reshape(1, -1)

        # Predict quality score using the loaded model
        quality_score = model.predict(input_data)[0]  # Raw prediction
        max_input = np.max(np.abs(input_data))
        quality_score = min(max(quality_score / max_input * 100 if max_input != 0 else quality_score * 100, 0), 100)
        print(f"Original predicted quality_score: {quality_score}")  # Log original scaled score
        
        # Temporary override for testing
        quality_score_override = 60  # Explicitly set override value
        print(f"Overridden quality_score: {quality_score_override}")  # Log overridden value
        quality_score = quality_score_override  # Apply override

        # Categorize quality
        quality_category = categorize_quality(quality_score)

        # Determine skin type and benefit using only oleic_acid
        skin_type, skin_benefit = determine_skin_type(data['oleic_acid'])
        print(f"Detected skin_type: {skin_type}, oleic_acid: {data['oleic_acid']}")  # Log decision factors

        # Calculate compound match
        pop_pos = data.get('pop', 0) + data.get('pos', 0)
        compound_match = calculate_compound_match(skin_type, data['oleic_acid'], data['sos'], pop_pos)

        # Determine recommendation level
        recommendation_level = determine_recommendation_level(quality_score, compound_match)

        # Generate skin concerns and recommendation note
        skin_concerns = generate_skin_concerns(skin_type)
        recommendation_note = generate_recommendation_note(quality_score, skin_type)

        # Generate recommended products
        recommended_products = generate_recommended_products(skin_type, quality_category)

        # Describe how cocoa butter works
        how_cocoa_works = how_cocoa_butter_works(skin_type, data['oleic_acid'], data['sos'])

        # Parameter analysis
        parameters = [
            {
                'parameter': 'Oleic Acid',
                'value': f"{data['oleic_acid']:.3f}%",
                'status': 'Optimal' if 26.09 <= data['oleic_acid'] <= 38.85 else 'Too High',
                'impact': 'Affects texture stability'
            },
            {
                'parameter': 'Viscosity',
                'value': f"{data['viscosity']:.3f} cP",
                'status': 'Optimal' if 240.74 <= data['viscosity'] <= 349.95 else 'Too High',
                'impact': 'May cause application issues'
            },
            {
                'parameter': 'Lead Content',
                'value': f"{data['lead']:.3f} ppm",
                'status': 'Optimal' if 0.01 <= data['lead'] <= 0.10 else 'Too High',
                'impact': 'Within safe limits'
            }
        ]

        # Scatter plot data (example: oleic_acid vs viscosity for skin types)
        scatter_data = [
            {'x': data['oleic_acid'], 'y': data['viscosity'], 'label': skin_type, 'color': '#FF6384'},
            {'x': 30, 'y': 300, 'label': 'normal', 'color': '#36A2EB'},
            {'x': 35, 'y': 320, 'label': 'oily', 'color': '#FFCE56'},
            {'x': 28, 'y': 250, 'label': 'dry', 'color': '#4BC0C0'},
            {'x': 32, 'y': 310, 'label': 'combination', 'color': '#9966FF'}
        ]

        # Prepare recommendations
        recommendations = {
            'category': quality_category,
            'description': f"This cocoa butter is {quality_category} quality. {'No refining needed' if quality_category == 'good' else 'Refining recommended'} for optimal use.",
            'refining_steps': refining_steps[quality_category],
            'products': products[quality_category]
        }

        return jsonify({
            'quality_score': quality_score,
            'quality_category': quality_category,
            'parameters': parameters,
            'recommendations': recommendations,
            'skin_type': skin_type,
            'skin_benefit': skin_benefit,
            'scatter_data': scatter_data,
            'compound_match': compound_match,
            'recommendation_level': recommendation_level,
            'skin_concerns': skin_concerns,
            'recommendation_note': recommendation_note,
            'recommended_products': recommended_products,
            'oleic_acid': data['oleic_acid'],
            'stearic_acid': data['stearic_acid'],
            'palmitic_acid': data['palmitic_acid'],
            'sos': data['sos'],
            'pop_pos': pop_pos,
            'how_cocoa_butter_works': how_cocoa_works
        })

    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)