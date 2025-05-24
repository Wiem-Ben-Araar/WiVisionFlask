import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.resolve()

class Config:
    # Core
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    ENV = os.environ.get('FLASK_ENV', 'production')
    
    # Paths
    UPLOAD_FOLDER = BASE_DIR / 'app' / 'uploads'
    CLASH_IMAGES_FOLDER = BASE_DIR / 'app' / 'static' / 'clash_images'
    REPORTS_FOLDER = BASE_DIR / 'app' / 'static' / 'reports'
    TEMPLATES_FOLDER = BASE_DIR / 'app' / 'templates'
    
    # Security
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'ifc', 'ifczip'}
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Static
    STATIC_URL_PATH = '/static'
    SERVE_STATIC = True
    
    # Environment specific
    if ENV == 'development':
        DEBUG = True
        TEMPLATES_AUTO_RELOAD = True
        SESSION_COOKIE_SECURE = False
    else:
        DEBUG = False

def create_folders(config):
    required_folders = [
        config.UPLOAD_FOLDER,
        config.CLASH_IMAGES_FOLDER,
        config.REPORTS_FOLDER,
        config.TEMPLATES_FOLDER
    ]
    
    for folder in required_folders:
        folder.mkdir(parents=True, exist_ok=True)
        folder.chmod(0o755)