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
    raise
except Exception as e:
    print(f"Error loading model with pickle: {e}. Please check the model file compatibility.")
    raise

# Define feature names matching the HTML input fields
feature_names = [
    'oleic_acid', 'stearic_acid', 'palmitic_acid', 'melting_point', 'solid_fat',
    'viscosity', 'moisture', 'sos', 'pop', 'pos', 'lead', 'cadmium',
    'refining_time', 'conching_time', 'temperature', 'nir', 'ftir', 'fermentation'
]

# Validation ranges (based on frontend input constraints)
validation_ranges = {
    'oleic_acid': (26.09, 38.85),
    'stearic_acid': (32.06, 41.00),
    'palmitic_acid': (24.00, 29.98),
    'melting_point': (31.01, 35.77),
    'solid_fat': (50.02, 64.71),
    'viscosity': (240.74, 349.95),
    'moisture': (0.20, 0.50),
    'sos': (0.30, 0.44),
    'pop': (0.25, 0.35),
    'pos': (0.22, 0.30),
    'lead': (0.01, 0.10),
    'cadmium': (0.003, 0.020),
    'refining_time': (10, 16),
    'conching_time': (4, 8),
    'temperature': (68.32, 77.77),
    'nir': (0.56, 0.90),
    'ftir': (0.50, 0.80)
}

# Function to categorize quality based on score (scaled to 0-100)
def categorize_quality(score):
    if score >= 80:
        return 'good'
    elif score >= 50:
        return 'medium'
    else:
        return 'low'

# Function to determine skin type based on quality score and key parameters
def determine_skin_type(quality_score, oleic_acid, viscosity):
    if quality_score >= 80 and 26.09 <= oleic_acid <= 33.0 and viscosity <= 300:
        return 'normal', 'Well-balanced hydration and texture suitable for normal skin.'
    elif quality_score >= 50 and oleic_acid > 33.0 and viscosity > 300:
        return 'oily', 'High oleic acid and viscosity may suit oily skin with proper refining.'
    elif quality_score < 50 or viscosity < 250:
        return 'dry', 'Lower viscosity and quality may help dry skin with added moisture.'
    else:
        return 'combination', 'Mixed properties suitable for combination skin with adjustments.'

# Function to calculate compound match percentage
def calculate_compound_match(skin_type, oleic_acid, viscosity, sos, pop_pos):
    base_match = 50
    if skin_type == 'normal' and 26.09 <= oleic_acid <= 33.0 and viscosity <= 300:
        base_match += 30
    elif skin_type == 'oily' and oleic_acid > 33.0 and viscosity > 300:
        base_match += 30
    elif skin_type == 'dry' and viscosity < 250:
        base_match += 30
    elif skin_type == 'combination':
        base_match += 20
    if 0.30 <= sos <= 0.44:  # Validate SOS range
        base_match += 10
    if 0.47 <= pop_pos <= 0.65:  # Validate POP + POS range
        base_match += 10
    return min(max(base_match, 0), 100)

# Function to determine recommendation level
def determine_recommendation_level(quality_score, skin_type_match):
    if quality_score >= 80 and skin_type_match >= 80:
        return 'High'
    elif quality_score >= 50 and skin_type_match >= 50:
        return 'Medium'
    else:
        return 'Low'

# Function to generate skin concerns
def generate_skin_concerns(skin_type):
    concerns = {
        'dry': 'Texture, Hydration',
        'oily': 'Oil Control, Pore Clogging',
        'normal': 'Maintain Balance',
        'combination': 'Zonal Hydration, Oil Control'
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
            {'name': 'Hydrating Body Butter', 'desc': 'Rich formula with high SOS and Oleic acid for deep moisturization', 'match': 92},
            {'name': 'Overnight Repair Cream', 'desc': 'Intensive repair with high Stearic Acid and POS for skin barrier', 'match': 87}
        ],
        'oily': [
            {'name': 'Matte Finish Lotion', 'desc': 'Light formula with POP/POS for oil control', 'match': 85},
            {'name': 'Oil-Free Balm', 'desc': 'Non-comedogenic with low Oleic Acid', 'match': 80}
        ],
        'normal': [
            {'name': 'Daily Moisturizer', 'desc': 'Balanced Oleic and Stearic Acid for everyday use', 'match': 90},
            {'name': 'Light Cream', 'desc': 'Gentle hydration with POP/POS', 'match': 88}
        ],
        'combination': [
            {'name': 'Zonal Hydration Gel', 'desc': 'Targets dry areas with SOS, controls oil with POP/POS', 'match': 83},
            {'name': 'Dual-Action Balm', 'desc': 'Hydrates dry zones, mattifies oily zones', 'match': 79}
        ]
    }
    return products_by_type.get(skin_type, [
        {'name': 'Lip Balm', 'desc': 'High-quality moisturization'} if quality_category == 'good' else
        {'name': 'Body Lotion', 'desc': 'Moisturizing with medium absorption'} if quality_category == 'medium' else
        {'name': 'Industrial Lubricant', 'desc': 'Non-cosmetic use'}
    ])

# Function to describe how cocoa butter works
def how_cocoa_butter_works(skin_type, oleic_acid,  sos):
    fatty_acids = f"Stearic and palmitic acids form a protective barrier, enhancing {skin_type} skin resilience with {oleic_acid:.1f}% Oleic Acid."
    triglycerides = f"SOS triglycerides at {sos:.1f}% create a moisture-locking matrix, ideal for {skin_type} skin hydration."
    antioxidants = "Natural polyphenols neutralize free radicals, reducing aging signs across all skin types."
    return {'fatty_acids': fatty_acids, 'triglycerides': triglycerides, 'antioxidants': antioxidants}

# Define refining steps based on quality category
refining_steps = {
    'good': ['No additional refining needed'],
    'medium': ['Filtration with activated carbon', 'Deodorization with steam', 'Winterization'],
    'low': ['Filtration with activated carbon', 'Deodorization with steam', 'Fractionation', 'Winterization']
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
    return render_template('skin.html')  # Corrected to use processing.html

# Route for additional page (index1.html)
@app.route('/index2')
def index1():
    return render_template('index2.html')

# Route for quality analysis using the ML model
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json

        # Extract and validate input data
        input_data = []
        for feature in feature_names:
            value = float(data.get(feature, 0))
            if feature in validation_ranges:
                min_val, max_val = validation_ranges[feature]
                if not (min_val <= value <= max_val):
                    return jsonify({'error': f'{feature} must be between {min_val} and {max_val}'}), 400
            input_data.append(value)

        # Convert to numpy array and reshape for prediction
        input_data = np.array(input_data).reshape(1, -1)

        # Predict quality score using the loaded model
        quality_score = model.predict(input_data)[0]
        # Normalize to 0-100 based on model output range (assuming model outputs 0-1, adjust if different)
        quality_score = min(max(quality_score * 100, 0), 100)  # Simple scaling, adjust based on model training

        # Categorize quality
        quality_category = categorize_quality(quality_score)

        # Determine skin type and benefit
        skin_type, skin_benefit = determine_skin_type(quality_score, data['oleic_acid'], data['viscosity'])

        # Calculate compound match
        compound_match = calculate_compound_match(skin_type, data['oleic_acid'], data['viscosity'], data['sos'], data['pop'] + data['pos'])

        # Determine recommendation level
        recommendation_level = determine_recommendation_level(quality_score, compound_match)

        # Generate skin concerns and recommendation note
        skin_concerns = generate_skin_concerns(skin_type)
        recommendation_note = generate_recommendation_note(quality_score, skin_type)

        # Generate recommended products
        recommended_products = generate_recommended_products(skin_type, quality_category)

        # Describe how cocoa butter works
        how_cocoa_works = how_cocoa_butter_works(skin_type, data['oleic_acid'], data['viscosity'], data['sos'])

        # Parameter analysis for all features
        parameters = []
        for feature in feature_names:
            value = data[feature]
            min_val, max_val = validation_ranges.get(feature, (0, float('inf')))
            status = 'Optimal' if min_val <= value <= max_val else 'Too High' if value > max_val else 'Too Low'
            impact = {
                'oleic_acid': 'Affects texture stability',
                'viscosity': 'May cause application issues',
                'lead': 'Within safe limits',
                'cadmium': 'Within safe limits'
            }.get(feature, 'No significant impact')
            parameters.append({
                'parameter': feature.replace('_', ' ').title(),
                'value': f"{value:.3f} {feature.split('_')[-1]}",
                'status': status,
                'impact': impact
            })

        # Prepare recommendations
        recommendations = {
            'category': quality_category,
            'description': f"This cocoa butter is {quality_category} quality. {'No refining needed' if quality_category == 'good' else 'Refining recommended'} for optimal use.",
            'refining_steps': refining_steps[quality_category],
            'products': recommended_products
        }

        return jsonify({
            'quality_score': quality_score,
            'quality_category': quality_category,
            'parameters': parameters,
            'recommendations': recommendations,
            'skin_type': skin_type,
            'skin_benefit': skin_benefit,
            'compound_match': compound_match,
            'recommendation_level': recommendation_level,
            'skin_concerns': skin_concerns,
            'recommendation_note': recommendation_note,
            'recommended_products': recommended_products,
            'how_cocoa_butter_works': how_cocoa_works
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)