import pandas as pd
import numpy as np
# Set matplotlib backend to 'Agg' to avoid "main thread is not in main loop" error
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import os
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Create models directory if it doesn't exist
os.makedirs('mobile_price_prediction/models/saved_models', exist_ok=True)

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath)
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # If there are missing values, handle them (but this dataset typically doesn't have any)
    df = df.dropna() if df.isnull().sum().sum() > 0 else df
    
    # Split features and target
    X = df.drop('price_range', axis=1) if 'price_range' in df.columns else df
    
    if 'price_range' in df.columns:
        y = df['price_range']
    else:
        y = None  # For prediction data without labels
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'mobile_price_prediction/models/saved_models/scaler.joblib')
    
    return X, X_scaled, y, df.columns.drop('price_range') if 'price_range' in df.columns else df.columns

def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors model"""
    print("Training KNN model...")
    # Set n_neighbors to min(5, len(X_train)) to ensure it works with small datasets
    n_neighbors = min(5, len(X_train))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def train_kmeans(X_train, y_train):
    """Train K-Means clustering model"""
    print("Training K-Means model...")
    kmeans = KMeans(n_clusters=4, random_state=42)  # Since we have 4 price ranges
    kmeans.fit(X_train)
    return kmeans

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model"""
    print("Training Naive Bayes model...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model"""
    print("Training Decision Tree model...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    return dt

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    # For K-Means, we need to map cluster labels to price ranges
    if model_name == 'K-Means':
        # Create a mapping from cluster label to most common price range
        mapping = {}
        for cluster in range(len(np.unique(y_pred))):
            mask = y_pred == cluster
            if np.any(mask):
                true_labels = y_test.iloc[mask] if hasattr(y_test, 'iloc') else y_test[mask]
                most_common = pd.Series(true_labels).mode()[0]
                mapping[cluster] = most_common
        
        # Map the cluster labels to price ranges
        y_pred_mapped = np.array([mapping.get(label, 0) for label in y_pred])
        y_pred = y_pred_mapped
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(classification_report(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred
    }

def visualize_features(X, y, feature_names):
    """Create PCA and t-SNE visualizations"""
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # t-SNE for dimensionality reduction
    # Dynamically adjust perplexity based on dataset size
    n_samples = X.shape[0]
    perplexity = min(30, max(5, n_samples // 5))  # Default is 30, but we adjust for small datasets
    
    # Skip t-SNE if dataset is too small (fewer than 10 samples)
    if n_samples >= 10:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA Plot
        scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
        axes[0].set_title('PCA Visualization')
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        plt.colorbar(scatter_pca, ax=axes[0], label='Price Range')
        
        # t-SNE Plot
        scatter_tsne = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8)
        axes[1].set_title(f't-SNE Visualization (perplexity={perplexity})')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter_tsne, ax=axes[1], label='Price Range')
    else:
        # Only create PCA plot if dataset is too small for t-SNE
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_pca = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
        ax.set_title('PCA Visualization')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        plt.colorbar(scatter_pca, ax=ax, label='Price Range')
    
    plt.tight_layout()
    plt.savefig('mobile_price_prediction/static/images/dimension_reduction.png', dpi=300, bbox_inches='tight')
    
    # Save PCA model
    joblib.dump(pca, 'mobile_price_prediction/models/saved_models/pca.joblib')
    
    # Feature Importance Analysis using PCA
    pca_full = PCA()
    pca_full.fit(X)
    
    # Variance explained by each component
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    ax.set_title('Explained Variance vs. Number of Components')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.grid(True)
    plt.savefig('mobile_price_prediction/static/images/pca_variance.png', dpi=300, bbox_inches='tight')
    
    # Get the top features based on PCA loadings
    components = pca_full.components_
    feature_importance = np.abs(components[0])  # Using the first principal component
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_features = [(feature_names[i], feature_importance[i]) for i in sorted_indices[:min(10, len(feature_names))]]
    
    # Create a bar plot of feature importances
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    ax.bar(names, values)
    ax.set_title('Top Features by PCA Loading (First Component)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Absolute Loading')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('mobile_price_prediction/static/images/feature_importance.png', dpi=300, bbox_inches='tight')
    
    return pca

def compare_models(evaluation_results):
    """Compare model performances and visualize them"""
    # Ensure we're starting with a clean slate by closing all figures
    plt.close('all')  
    
    # Extract metrics
    models = [result['model_name'] for result in evaluation_results]
    accuracy = [result['accuracy'] for result in evaluation_results]
    precision = [result['precision'] for result in evaluation_results]
    recall = [result['recall'] for result in evaluation_results]
    f1 = [result['f1_score'] for result in evaluation_results]
    
    # Create a comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    
    # Identify the best model based on F1 score
    best_model_idx = comparison_df['F1 Score'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]
    
    print(f"\nModel Comparison:\n{comparison_df}")
    print(f"\nBest Model: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.4f}")
    
    # Debug: Print all values to ensure they're not empty
    print(f"Debug - Models: {models}")
    print(f"Debug - Accuracy: {accuracy}")
    print(f"Debug - Precision: {precision}")
    print(f"Debug - Recall: {recall}")
    print(f"Debug - F1 Score: {f1}")
    
    # Create a figure explicitly with Figure instead of plt.figure
    fig = Figure(figsize=(12, 8), facecolor='white')
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Set up bar positions
    bar_width = 0.2
    x = np.arange(len(models))
    
    # Plot bars for each metric with strong colors and ensure they're visible
    bars1 = ax.bar(x - bar_width*1.5, accuracy, width=bar_width, label='Accuracy', color='#1f77b4', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x - bar_width*0.5, precision, width=bar_width, label='Precision', color='#ff7f0e', edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + bar_width*0.5, recall, width=bar_width, label='Recall', color='#2ca02c', edgecolor='black', linewidth=1)
    bars4 = ax.bar(x + bar_width*1.5, f1, width=bar_width, label='F1 Score', color='#d62728', edgecolor='black', linewidth=1)
    
    # Add value labels on top of each bar
    for i, v in enumerate(accuracy):
        ax.text(i - bar_width*1.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(precision):
        ax.text(i - bar_width*0.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(recall):
        ax.text(i + bar_width*0.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(f1):
        ax.text(i + bar_width*1.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    # Add a horizontal line for average performance
    avg_performance = np.mean([accuracy, precision, recall, f1])
    ax.axhline(y=avg_performance, color='gray', linestyle='--', alpha=0.7, label='Average')
    
    # Add labels, title and legend
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontweight='bold')
    ax.set_ylim(0, 1.05)  # Set y-axis limit slightly above 1 to show value labels
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight the best model
    best_model_index = models.index(best_model['Model'])
    ax.annotate('Best Model', 
                xy=(best_model_index, 0.1),
                xytext=(best_model_index, 0.05),
                ha='center',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    # Add decorative frame
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Ensure tight layout
    fig.tight_layout()
    
    # Make sure the figure is drawn
    canvas.draw()
    
    # Explicitly save as PNG with high resolution
    fig.savefig('mobile_price_prediction/static/images/model_comparison.png', 
               dpi=300, 
               bbox_inches='tight', 
               format='png',
               facecolor=fig.get_facecolor())
    
    print(f"Model comparison plot saved to: mobile_price_prediction/static/images/model_comparison.png")
    
    return comparison_df, best_model['Model']

def get_feature_importances(model, feature_names, model_name):
    """Extract feature importances for tree-based models"""
    if model_name == 'Decision Tree':
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances ({model_name})')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('mobile_price_prediction/static/images/dt_feature_importance.png', dpi=300, bbox_inches='tight')

def train_and_evaluate_models():
    """Main function to train and evaluate all models"""
    # Create static/images directory if it doesn't exist
    os.makedirs('mobile_price_prediction/static/images', exist_ok=True)
    
    # Check if the training file exists or has sufficient data
    train_path = 'mobile_price_prediction/data/train.csv'
    if not os.path.exists(train_path) or os.path.getsize(train_path) < 1000:  # We expect more than 1KB of data
        print("Generating sample training data as the existing data is insufficient...")
        
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
    
    # Load and preprocess data
    X, X_scaled, y, feature_names = load_and_preprocess_data(train_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train models
    knn_model = train_knn(X_train, y_train)
    kmeans_model = train_kmeans(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    
    # Save models
    joblib.dump(knn_model, 'mobile_price_prediction/models/saved_models/knn_model.joblib')
    joblib.dump(kmeans_model, 'mobile_price_prediction/models/saved_models/kmeans_model.joblib')
    joblib.dump(nb_model, 'mobile_price_prediction/models/saved_models/nb_model.joblib')
    joblib.dump(dt_model, 'mobile_price_prediction/models/saved_models/dt_model.joblib')
    
    # Evaluate models
    results = []
    results.append(evaluate_model(knn_model, X_test, y_test, 'KNN'))
    results.append(evaluate_model(kmeans_model, X_test, y_test, 'K-Means'))
    results.append(evaluate_model(nb_model, X_test, y_test, 'Naive Bayes'))
    results.append(evaluate_model(dt_model, X_test, y_test, 'Decision Tree'))
    
    # Get feature importances for Decision Tree
    get_feature_importances(dt_model, feature_names, 'Decision Tree')
    
    # Compare model performances
    comparison_df, best_model = compare_models(results)
    
    # Create PCA and t-SNE visualizations
    visualize_features(X_scaled, y, feature_names)
    
    return comparison_df, best_model

if __name__ == "__main__":
    comparison_df, best_model = train_and_evaluate_models()
    print(f"\nModel training and evaluation completed. Best model: {best_model}")