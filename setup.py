from setuptools import setup, find_packages

setup(
    name="mobile_price_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
    ],
    python_requires=">=3.6",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Flask application for predicting mobile phone price ranges",
    keywords="machine learning, flask, mobile price prediction",
    url="https://github.com/yourusername/mobile-price-prediction",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)