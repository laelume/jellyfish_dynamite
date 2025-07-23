# config.py

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'jellyfish-dynamite-key'
    UPLOAD_FOLDER = 'temp_uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # Analysis defaults
    DEFAULT_METHODS = ['FFT', 'CQT', 'Multi-Res', 'Chirplet Zero']
    DEFAULT_N_FFT = 1024
    DEFAULT_PEAK_FMIN = 100
    DEFAULT_PEAK_FMAX = 6000
    
    # Performance settings
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_FILES_PER_SESSION = 100