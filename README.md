# Mobile Price Range Prediction

A complete machine learning project to predict mobile phone price ranges based on specifications using various algorithms.

## Project Overview

This project implements multiple machine learning algorithms to predict the price range of mobile phones based on their specifications. It features a Flask web application that provides an easy-to-use interface for:

1. Training and evaluating machine learning models
2. Making individual predictions through a web form
3. Processing batch predictions via CSV file uploads
4. Visualizing model performance and feature importance

## Features

- **Data Preprocessing**: Handles missing values and scales features
- **Multiple Algorithms**: Implements 4 different approaches:
  - K-Nearest Neighbors (KNN)
  - K-Means Clustering
  - Naive Bayes
  - Decision Tree
- **Comprehensive Model Evaluation**: Compares models using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- **Advanced Visualizations**:
  - Model performance comparison
  - PCA and t-SNE dimensionality reduction
  - Feature importance analysis
- **Production-Ready Web Application**:
  - Interactive Bootstrap UI
  - Single prediction form input
  - Batch prediction via CSV uploads
  - Detailed results and visualizations

## Dataset Information

The project uses the [Mobile Price Classification dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) from Kaggle, which includes the following features:

- `battery_power`: Total energy a battery can store (in mAh)
- `blue`: Has Bluetooth or not (0/1)
- `clock_speed`: Speed of processor
- `dual_sim`: Has dual SIM support (0/1)
- `fc`: Front camera megapixels
- `four_g`: Has 4G (0/1)
- `int_memory`: Internal memory in GB
- `m_dep`: Mobile depth in cm
- `mobile_wt`: Weight of mobile phone
- `n_cores`: Number of processor cores
- `pc`: Primary camera megapixels
- `px_height`: Pixel resolution height
- `px_width`: Pixel resolution width
- `ram`: RAM in MB
- `sc_h`: Screen height in cm
- `sc_w`: Screen width in cm
- `talk_time`: Longest time a battery charge will last
- `three_g`: Has 3G (0/1)
- `touch_screen`: Has touch screen (0/1)
- `wifi`: Has wifi (0/1)
- `price_range`: Target variable with 4 price categories (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)

## Project Structure

```
mobile_price_prediction/
├── data/                 # Directory for storing CSV data files
├── models/               # ML model implementation
│   ├── saved_models/     # Directory for saved trained models
│   └── train_models.py   # Script for model training and evaluation
├── routes/               # Flask route definitions
│   └── model_routes.py   # Blueprint for all model-related routes
├── static/               # Static assets
│   ├── css/              # CSS stylesheets
│   │   └── style.css     # Custom application styling
│   ├── js/               # JavaScript files
│   └── images/           # Generated visualizations and images
└── templates/            # HTML templates
    ├── base.html         # Base template with layout
    ├── index.html        # Home page
    ├── predict.html      # Single prediction form
    ├── result.html       # Single prediction results
    ├── upload.html       # Batch prediction file upload
    ├── batch_result.html # Batch prediction results
    ├── train.html        # Model training page
    ├── training_result.html # Training results
    └── error.html        # Error page
main.py                   # Main application entry point
requirements.txt          # Project dependencies
README.md                 # Project documentation
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mobile-price-prediction.git
cd mobile-price-prediction
```

2. Create a virtual environment and activate it:

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python main.py
```

5. Open your browser and go to `http://127.0.0.1:5000/`

## Usage Guide

### 1. Training Models

1. Download the [Mobile Price Classification dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) from Kaggle
2. Go to the "Train Models" page
3. Upload the training CSV file (must contain the `price_range` column)
4. View the model performance comparison and visualizations
5. The best model will be automatically selected for future predictions

### 2. Making Individual Predictions

1. Go to the "Predict" page
2. Enter the specifications for a mobile phone
3. Click "Make Prediction" to see the predicted price range from all models
4. View the recommended price range from the best-performing model

### 3. Making Batch Predictions

1. Go to the "Batch Predict" page
2. Upload a CSV file containing specifications for multiple mobile phones
3. View the prediction summary and download the complete results

## Model Comparison and Selection

This project automatically compares the performance of all four models and selects the best one based on F1 score. The comparison results are displayed visually, and the best model is used for making "recommended" predictions.

## Visualization and Interpretability

The application provides several visualizations to help understand:
- Model performance comparison
- Feature importance
- Data distribution in lower dimensions (PCA and t-SNE)
- Explained variance in PCA components

These visualizations help in interpreting the models and making informed decisions based on the predictions.

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization
- **Bootstrap**: Frontend styling

## Future Improvements

- Add more advanced algorithms (e.g., Random Forest, Gradient Boosting)
- Implement hyperparameter tuning
- Add user authentication system
- Create an API endpoint for predictions
- Implement model retraining with new data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original dataset from [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- Inspired by real-world applications of machine learning in e-commerce