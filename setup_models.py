import pandas as pd
import numpy as np
import os
import sys

# Add the project root to system path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

print("Setting up model training environment...")

# Ensure data directory exists
os.makedirs('mobile_price_prediction/data', exist_ok=True)
os.makedirs('mobile_price_prediction/models/saved_models', exist_ok=True)
os.makedirs('mobile_price_prediction/static/images', exist_ok=True)

# Check if train.csv exists and has valid data
train_path = 'mobile_price_prediction/data/train.csv'

if not os.path.exists(train_path) or os.path.getsize(train_path) < 100:
    # Generate sample data if train.csv doesn't exist or is too small
    print("Generating sample training data...")
    
    # Create sample data with 200 records
    n_samples = 200
    np.random.seed(42)  # For reproducibility
    
    # Generate features with reasonable correlations to price
    battery_power = np.random.randint(500, 2000, n_samples)
    blue = np.random.randint(0, 2, n_samples)
    clock_speed = np.random.uniform(0.5, 3.0, n_samples)
    dual_sim = np.random.randint(0, 2, n_samples)
    fc = np.random.randint(0, 20, n_samples)
    four_g = np.random.randint(0, 2, n_samples)
    int_memory = np.random.randint(2, 64, n_samples)
    m_dep = np.random.uniform(0.1, 1.0, n_samples)
    mobile_wt = np.random.randint(80, 200, n_samples)
    n_cores = np.random.randint(1, 8, n_samples)
    pc = np.random.randint(2, 20, n_samples)
    px_height = np.random.randint(500, 2000, n_samples)
    px_width = np.random.randint(500, 2000, n_samples)
    ram = np.random.randint(256, 4096, n_samples)
    sc_h = np.random.randint(5, 20, n_samples)
    sc_w = np.random.randint(5, 15, n_samples)
    talk_time = np.random.randint(5, 20, n_samples)
    three_g = np.random.randint(0, 2, n_samples)
    touch_screen = np.random.randint(0, 2, n_samples)
    wifi = np.random.randint(0, 2, n_samples)
    
    # Create price range based on features (simple model to ensure correlation)
    # Higher RAM, more storage, better camera = higher price range
    price_scores = (
        0.3 * (ram / 4096) + 
        0.2 * (int_memory / 64) + 
        0.2 * (pc / 20) +
        0.1 * (battery_power / 2000) +
        0.1 * (n_cores / 8) +
        0.1 * (clock_speed / 3.0)
    )
    
    # Convert scores to price ranges (0-3)
    price_range = np.digitize(price_scores, [0.25, 0.5, 0.75]) - 1
    
    features = {
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi,
        'price_range': price_range
    }
    
    # Create DataFrame and save to CSV
    train_df = pd.DataFrame(features)
    train_df.to_csv(train_path, index=False)
    print(f"Sample training data saved to {train_path}")
else:
    print(f"Using existing training data from {train_path}")

# Also create a test data file for predictions
test_path = 'mobile_price_prediction/data/test.csv'

if not os.path.exists(test_path) or os.path.getsize(test_path) < 100:
    print("Generating sample test data...")
    
    # Create sample test data with 50 records
    n_test_samples = 50
    np.random.seed(43)  # Different seed for test data
    
    test_features = {
        'battery_power': np.random.randint(500, 2000, n_test_samples),
        'blue': np.random.randint(0, 2, n_test_samples),
        'clock_speed': np.random.uniform(0.5, 3.0, n_test_samples),
        'dual_sim': np.random.randint(0, 2, n_test_samples),
        'fc': np.random.randint(0, 20, n_test_samples),
        'four_g': np.random.randint(0, 2, n_test_samples),
        'int_memory': np.random.randint(2, 64, n_test_samples),
        'm_dep': np.random.uniform(0.1, 1.0, n_test_samples),
        'mobile_wt': np.random.randint(80, 200, n_test_samples),
        'n_cores': np.random.randint(1, 8, n_test_samples),
        'pc': np.random.randint(2, 20, n_test_samples),
        'px_height': np.random.randint(500, 2000, n_test_samples),
        'px_width': np.random.randint(500, 2000, n_test_samples),
        'ram': np.random.randint(256, 4096, n_test_samples),
        'sc_h': np.random.randint(5, 20, n_test_samples),
        'sc_w': np.random.randint(5, 15, n_test_samples),
        'talk_time': np.random.randint(5, 20, n_test_samples),
        'three_g': np.random.randint(0, 2, n_test_samples),
        'touch_screen': np.random.randint(0, 2, n_test_samples),
        'wifi': np.random.randint(0, 2, n_test_samples)
    }
    
    # Create test DataFrame (no price_range as this is for prediction)
    test_df = pd.DataFrame(test_features)
    test_df.to_csv(test_path, index=False)
    print(f"Sample test data saved to {test_path}")
else:
    print(f"Using existing test data from {test_path}")

# Train models using the data
print("\nTraining models...")
from mobile_price_prediction.models.train_models import train_and_evaluate_models
comparison_df, best_model = train_and_evaluate_models()
print(f"\nModel training completed. Best model: {best_model}")
print(f"\nModel performance metrics:")
print(comparison_df)

print("\nSetup completed successfully! Your models are now trained and ready to use.")
print("You can now run the Flask application by executing: python main.py")