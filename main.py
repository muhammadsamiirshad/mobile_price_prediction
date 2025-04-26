from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump, load
import io
import base64

# Import routes
from mobile_price_prediction.routes.model_routes import model_bp

app = Flask(__name__, 
            static_folder='mobile_price_prediction/static',
            template_folder='mobile_price_prediction/templates')

# Register blueprints
app.register_blueprint(model_bp)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)