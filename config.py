# config.py

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'jellyfish-dynamite-key'
    UPLOAD_FOLDER = 'temp_uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # Analysis defaults - CHANGED: FFT_DUAL as default
    DEFAULT_METHODS = ['FFT_DUAL']  # Only FFT_DUAL checked by default
    ALL_METHODS = ['FFT_DUAL', 'CQT', 'Multi-Res', 'Chirplet Zero']  # All available methods
    
    # Resolution options - NEW
    PSD_NFFT_OPTIONS = [256, 512, 1024, 2048, 4096, 8192]
    SPEC_NFFT_OPTIONS = [128, 256, 512, 1024, 2048, 4096]
    HOP_RATIO_OPTIONS = [2, 4, 6, 8]  # n_fft divided by these values
    
    DEFAULT_N_FFT = 1024
    DEFAULT_SPEC_N_FFT = 512
    DEFAULT_HOP_RATIO = 4
    DEFAULT_PEAK_FMIN = 100
    DEFAULT_PEAK_FMAX = 6000
    
    # Performance settings
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_FILES_PER_SESSION = 100