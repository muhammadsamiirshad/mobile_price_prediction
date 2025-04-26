import os

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    STATIC_FOLDER = 'mobile_price_prediction/static'
    TEMPLATES_FOLDER = 'mobile_price_prediction/templates'
    UPLOAD_FOLDER = 'mobile_price_prediction/data/uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # In production, use a proper secret key
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-key-change-me'

# Configuration dictionary to easily switch between environments
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}