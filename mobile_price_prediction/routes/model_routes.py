from flask import Blueprint, render_template, request, redirect, url_for, jsonify, flash
import os
import pandas as pd
import numpy as np
import joblib
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename

# Create a Blueprint for model routes
model_bp = Blueprint('model_bp', __name__)

# Create data directory if it doesn't exist
os.makedirs('mobile_price_prediction/data', exist_ok=True)

# Create allowed file extensions and upload folder
ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = 'mobile_price_prediction/data'

def allowed_file(filename):
    """Check if filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load all trained models"""
    models = {}
    model_path = 'mobile_price_prediction/models/saved_models'
    
    # Check if models exist, if not return empty dict
    if not os.path.exists(model_path):
        return models
    
    # Load models if they exist
    if os.path.exists(os.path.join(model_path, 'knn_model.joblib')):
        models['KNN'] = joblib.load(os.path.join(model_path, 'knn_model.joblib'))
    
    if os.path.exists(os.path.join(model_path, 'kmeans_model.joblib')):
        models['K-Means'] = joblib.load(os.path.join(model_path, 'kmeans_model.joblib'))
    
    if os.path.exists(os.path.join(model_path, 'nb_model.joblib')):
        models['Naive Bayes'] = joblib.load(os.path.join(model_path, 'nb_model.joblib'))
    
    if os.path.exists(os.path.join(model_path, 'dt_model.joblib')):
        models['Decision Tree'] = joblib.load(os.path.join(model_path, 'dt_model.joblib'))
    
    return models

def get_best_model():
    """Get name of best performing model based on saved results"""
    models = load_models()
    # Default to Decision Tree if no preference exists
    return "Decision Tree" if "Decision Tree" in models else list(models.keys())[0] if models else None

def make_prediction(features, model_names=None):
    """Make predictions using specified models"""
    models = load_models()
    
    if not models:
        return {"error": "No trained models found. Please train models first."}
    
    # Load the scaler
    scaler_path = 'mobile_price_prediction/models/saved_models/scaler.joblib'
    if not os.path.exists(scaler_path):
        return {"error": "Scaler not found. Please train models first."}
    
    scaler = joblib.load(scaler_path)

    try:
        # Check the training data range to detect extreme values
        train_data_path = 'mobile_price_prediction/data/train.csv'
        if os.path.exists(train_data_path):
            train_df = pd.read_csv(train_data_path)
            
            # Create a copy of features to avoid modifying the original
            features_copy = features.copy()
            
            # Log extreme values for debugging
            extreme_values = {}
            
            # For each feature, check if values are outside the training data range
            for col in features.columns:
                if col in train_df.columns:
                    min_val = train_df[col].min()
                    max_val = train_df[col].max()
                    
                    # Check each value in the feature
                    for i, val in enumerate(features[col]):
                        if val < min_val * 0.5 or val > max_val * 1.5:
                            extreme_values[col] = {'value': val, 'train_min': min_val, 'train_max': max_val}
                            # Don't modify the value, but log it for reference
        
        # Preprocess features - the scaler will handle the transformation
        features_scaled = scaler.transform(features)
        
        # If no model names are specified, use all available models
        if model_names is None or len(model_names) == 0:
            model_names = models.keys()
        
        predictions = {}
        kmeans_clusters_map = None
        
        # Make predictions with each model
        for name in model_names:
            if name in models:
                model = models[name]
                
                if name == 'K-Means':
                    # For K-Means, we need to map cluster labels to price ranges
                    if kmeans_clusters_map is None:
                        # Load training data to create mapping if not provided
                        # This is simplified - ideally, you'd save this mapping during training
                        clusters = model.predict(features_scaled)
                        # Default mapping: map each cluster to itself (0->0, 1->1, etc.)
                        predictions[name] = clusters.tolist()
                    else:
                        clusters = model.predict(features_scaled)
                        mapped_predictions = [kmeans_clusters_map.get(c, c) for c in clusters]
                        predictions[name] = mapped_predictions
                else:
                    # For classification models
                    preds = model.predict(features_scaled)
                    predictions[name] = preds.tolist()
        
        # Get best model prediction
        best_model = get_best_model()
        if best_model and best_model in predictions:
            predictions['recommended'] = predictions[best_model]
        else:
            # If no best model defined, use the first available
            first_model = list(predictions.keys())[0] if predictions else None
            if first_model:
                predictions['recommended'] = predictions[first_model]
        
        # Add warning for extreme values if any were found
        if extreme_values:
            predictions['warnings'] = extreme_values
        
        return predictions
        
    except Exception as e:
        # Return more detailed error information
        import traceback
        return {
            "error": f"Prediction failed: {str(e)}",
            "details": traceback.format_exc()
        }

def estimate_price_from_features(features, price_range):
    """
    Calculate an estimated price range based on device specifications and predicted price range.
    
    Args:
        features (dict): Input features of the mobile phone
        price_range (int or str): Predicted price range category (0-3)
    
    Returns:
        dict: Dictionary containing min_price, max_price, and exact_price
    """
    # Convert price range to int if it's a string or extract first element if it's a list/array
    if isinstance(price_range, (list, np.ndarray)):
        price_range = price_range[0]  # Take the first element if it's a list or array
    
    if isinstance(price_range, str):
        # Extract number from string like "Low Cost (0)"
        try:
            price_range = int(price_range.split("(")[1].split(")")[0])
        except (IndexError, ValueError):
            # Default to price_range 1 if parsing fails
            price_range = 1
    elif isinstance(price_range, dict):
        # If somehow a dictionary was passed, use a default value
        price_range = 1
    
    # Ensure price_range is an integer between 0 and 3
    try:
        price_range = int(price_range)
        price_range = max(0, min(3, price_range))  # Clamp between 0 and 3
    except (TypeError, ValueError):
        # Default to price_range 1 if conversion fails
        price_range = 1
    
    # Base price ranges (in PKR)
    base_ranges = {
        0: (8000, 15000),    # Low-cost range
        1: (15000, 25000),   # Budget range
        2: (25000, 45000),   # Mid-range
        3: (45000, 200000)   # Premium range
    }
    
    # Get base range
    min_price, max_price = base_ranges.get(price_range, (15000, 45000))
    
    # Calculate price adjustments based on key specs without limitations
    price_adjustments = 0
    
    # RAM impact on price (25% weight) - removed normalization cap
    ram_adjustment = (max_price - min_price) * 0.25 * (features['ram'] / 8000)
    
    # Internal memory impacts price (15% weight) - removed normalization cap
    storage_adjustment = (max_price - min_price) * 0.15 * (features['int_memory'] / 64)
    
    # Battery power (15% weight) - removed normalization cap
    battery_adjustment = (max_price - min_price) * 0.15 * (features['battery_power'] / 2000)
    
    # Camera quality (front + primary) (25% weight)
    # Primary camera (15%) - removed normalization cap
    pc_adjustment = (max_price - min_price) * 0.15 * (features['pc'] / 20)
    
    # Front camera (10%) - removed normalization cap
    fc_adjustment = (max_price - min_price) * 0.10 * (features['fc'] / 10)
    
    # Other features (20% weight):
    clock_adjustment = (max_price - min_price) * 0.04 * (features['clock_speed'] / 3.0)
    cores_adjustment = (max_price - min_price) * 0.04 * (features['n_cores'] / 8)
    
    # Weight factor - retain some reasonableness for weight
    weight_factor = max(0.0, 1.0 - (features['mobile_wt'] / 200)) 
    weight_adjustment = (max_price - min_price) * 0.04 * weight_factor
    
    # Screen and pixels - removed normalization caps
    screen_factor = (features['sc_h'] * features['sc_w']) / (1500 * 20)
    screen_adjustment = (max_price - min_price) * 0.04 * screen_factor
    
    pixel_factor = (features['px_height'] * features['px_width']) / (1920 * 1080)
    pixel_adjustment = (max_price - min_price) * 0.04 * pixel_factor
    
    # Apply all adjustments
    total_adjustment = ram_adjustment + storage_adjustment + battery_adjustment + pc_adjustment + \
                       fc_adjustment + clock_adjustment + cores_adjustment + weight_adjustment + \
                       screen_adjustment + pixel_adjustment
    
    # Calculate exact price
    exact_price = min_price + total_adjustment
    
    # Add some randomness (±5%) to make predictions less uniform
    randomness = 0.05 * exact_price  # 5% of exact price
    min_rand_price = int(max(min_price, exact_price - randomness))
    max_rand_price = int(exact_price + randomness)  # Removed max_price cap
    
    return {
        'min_price': min_rand_price,
        'max_price': max_rand_price,
        'exact_price': int(exact_price)
    }

@model_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle predictions via form submission"""
    if request.method == 'GET':
        # Load feature names from the training set if available
        feature_names = []
        try:
            if os.path.exists('mobile_price_prediction/data/train.csv'):
                df = pd.read_csv('mobile_price_prediction/data/train.csv')
                feature_names = [col for col in df.columns if col != 'price_range']
        except Exception as e:
            feature_names = [
                'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
                'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen',
                'wifi'
            ]
        
        return render_template('predict.html', feature_names=feature_names)
    
    elif request.method == 'POST':
        # Required features from the training data
        required_features = [
            'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
            'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
            'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen',
            'wifi'
        ]
        
        features = {}
        
        # Process form input and ensure all required features are present
        for feature in required_features:
            if feature in request.form and request.form[feature].strip():
                try:
                    features[feature] = float(request.form[feature])
                except ValueError:
                    return render_template('error.html', 
                                          message=f"Invalid value for {feature}: {request.form[feature]}. Please enter numeric values.")
            else:
                return render_template('error.html', 
                                      message=f"Missing required feature: {feature}. Please fill out all fields.")
        
        # Convert to DataFrame for prediction
        df = pd.DataFrame([features])
        
        # Ensure DataFrame has columns in the correct order
        df = df[required_features]
        
        # Make predictions
        try:
            predictions = make_prediction(df)
            if "error" in predictions:
                return render_template('error.html', message=predictions["error"])
            
            # Map predictions to friendly names
            price_range_names = {
                0: "Low Cost (0)",
                1: "Medium Cost (1)",
                2: "High Cost (2)",
                3: "Very High Cost (3)"
            }
            
            formatted_predictions = {}
            for model_name, preds in predictions.items():
                if model_name == 'warnings':
                    formatted_predictions[model_name] = preds  # Keep warnings as is
                else:
                    formatted_predictions[model_name] = [price_range_names.get(int(p), f"Unknown ({p})") for p in preds]
            
            # Generate dynamic price estimates for each model's prediction
            price_estimates = {}
            price_estimates_by_model = {}
            
            # First for the recommended model
            if 'recommended' in predictions:
                recommended_price_range = predictions['recommended'][0]
                price_estimates = estimate_price_from_features(features, recommended_price_range)
                formatted_predictions['price_estimate'] = price_estimates
            
            # Then for each model
            for model_name, preds in predictions.items():
                if model_name not in ['warnings', 'recommended']:
                    model_price_range = preds[0]
                    price_estimates_by_model[model_name] = estimate_price_from_features(features, model_price_range)
            
            formatted_predictions['price_estimates_by_model'] = price_estimates_by_model
            
            # Get feature importances and visualizations if available
            visualizations = {}
            if os.path.exists('mobile_price_prediction/static/images'):
                image_files = [f for f in os.listdir('mobile_price_prediction/static/images') 
                              if f.endswith('.png')]
                visualizations = {os.path.splitext(f)[0]: f'/static/images/{f}' for f in image_files}
            
            return render_template('result.html', 
                                  predictions=formatted_predictions,
                                  features=features,
                                  visualizations=visualizations)
            
        except Exception as e:
            return render_template('error.html', 
                                  message=f"Prediction error: {str(e)}")

@model_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload for batch prediction"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    elif request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('error.html', message="No file part")
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return render_template('error.html', message="No file selected")
        
        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            test_file_path = os.path.join(UPLOAD_FOLDER, 'test.csv')
            file.save(test_file_path)
            
            try:
                # Load the file
                df = pd.read_csv(test_file_path)
                
                # Check if the file has the right format
                if df.empty:
                    return render_template('error.html', message="Uploaded file is empty")
                
                # Make predictions
                predictions = make_prediction(df)
                if "error" in predictions:
                    return render_template('error.html', message=predictions["error"])
                
                # Prepare results for display
                results = df.copy()
                
                # Add predictions from each model
                for model_name, preds in predictions.items():
                    results[f"prediction_{model_name.replace(' ', '_').lower()}"] = preds
                
                # Save results to CSV for download
                result_file_path = os.path.join(UPLOAD_FOLDER, 'prediction_results.csv')
                results.to_csv(result_file_path, index=False)
                
                # Count predictions by class for each model
                summary = {}
                for model_name, preds in predictions.items():
                    unique, counts = np.unique(preds, return_counts=True)
                    summary[model_name] = {int(u): int(c) for u, c in zip(unique, counts)}
                
                # Get visualization images if available
                visualizations = {}
                if os.path.exists('mobile_price_prediction/static/images'):
                    image_files = [f for f in os.listdir('mobile_price_prediction/static/images') 
                                 if f.endswith('.png')]
                    visualizations = {os.path.splitext(f)[0]: f'/static/images/{f}' for f in image_files}
                
                return render_template('batch_result.html',
                                      file_name=filename,
                                      summary=summary,
                                      result_file=result_file_path.replace('mobile_price_prediction/', ''),
                                      visualizations=visualizations)
                
            except Exception as e:
                return render_template('error.html', 
                                     message=f"Error processing file: {str(e)}")
        
        return render_template('error.html', 
                             message="Invalid file type. Please upload a CSV file.")

@model_bp.route('/train', methods=['GET', 'POST'])
def train_models():
    """Train and evaluate models"""
    if request.method == 'GET':
        return render_template('train.html')
    
    elif request.method == 'POST':
        # Check if a training file was uploaded
        if 'file' not in request.files:
            return render_template('error.html', message="No file part")
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return render_template('error.html', message="No file selected")
        
        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, 'train.csv')
            file.save(file_path)
            
            try:
                # Import here to avoid circular imports
                from mobile_price_prediction.models.train_models import train_and_evaluate_models
                
                # Train models
                comparison_df, best_model = train_and_evaluate_models()
                
                # Convert DataFrame to dictionary for proper display
                comparison_data = comparison_df.to_dict('records')
                
                # Format decimal values to 2 decimal places for better display
                for record in comparison_data:
                    for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                        if key in record:
                            record[key] = round(record[key], 2)
                
                # Create HTML table with formatted values 
                comparison_html = pd.DataFrame(comparison_data).to_html(
                    classes='table table-striped table-bordered',
                    float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x
                )
                
                # Get visualization images
                visualizations = {}
                if os.path.exists('mobile_price_prediction/static/images'):
                    image_files = [f for f in os.listdir('mobile_price_prediction/static/images') 
                                 if f.endswith('.png')]
                    visualizations = {os.path.splitext(f)[0]: f'/static/images/{f}' for f in image_files}
                
                return render_template('training_result.html',
                                      comparison_table=comparison_html,
                                      best_model=best_model,
                                      visualizations=visualizations)
                
            except Exception as e:
                return render_template('error.html', 
                                     message=f"Error training models: {str(e)}")
        
        return render_template('error.html', 
                             message="Invalid file type. Please upload a CSV file.")