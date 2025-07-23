# jelly_funcs.py (constructed from all_functions_psd_rebuilt.py)

from datetime import datetime
import os
from pathlib import Path
import re
import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal of length=.*")

def get_subdir_pathlist(some_directory):
    subdir_pathlist = []
    for x in os.walk(some_directory): 
        subdir_path = x[0]
        #print(subdir_path)
        match = re.search(r'(\d+)$',subdir_path)
        if match: 
            #print(subdir_path)
            subdir_pathlist.append(subdir_path)
    return subdir_pathlist

def make_daily_directory(base_dir=None, create=True, verbose=True):
    """
    Always returns the same daily directory for calls on the same day,
    based on the 'code/artifacts' directory found upward from the current location - cwd.
    """
    if base_dir is None:
        base_dir = find_artifacts_dir()
    base_path = Path(base_dir)
    today = datetime.now().strftime("%Y-%m-%d")
    daily_dir = base_path / today
    if create and not daily_dir.exists():
        daily_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created daily directory: {daily_dir.absolute()}")
    elif verbose and daily_dir.exists():
        print(f"Using existing daily directory: {daily_dir.absolute()}")
    return daily_dir

def get_timestamp():
    return datetime.now().strftime("%H%M%S%f")[:-4]


def calculate_psd(audio_path, n_fft=1024, fmin=None, fmax=None):
    """Calculate PSD with zero padding, n_fft, frequency range - robust path handling."""
    
    # Convert to Path object for robust handling
    audio_path = Path(audio_path)
    
    # Check if file exists
    if not audio_path.exists():
        # Try .wav extension if not present
        if audio_path.suffix.lower() != '.wav':
            wav_path = audio_path.with_suffix('.wav')
            if wav_path.exists():
                audio_path = wav_path
            else:
                raise FileNotFoundError(f"WAV file not found: {audio_path} or {wav_path}")
        else:
            raise FileNotFoundError(f"WAV file not found: {audio_path}")
    
    # Convert back to string for librosa
    audio_path_str = str(audio_path)
    
    try:
        # Load audio file and let librosa determine the sample rate
        y, sr = librosa.load(audio_path_str, sr=None)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path_str}: {str(e)}")
    
    if len(y) == 0:
        raise ValueError(f"Audio file is empty: {audio_path_str}")
    
    hop_length = 512
    
    # Compute STFT with zero-padding
    stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    power_spectrum = np.abs(stft_result)**2
    psd_mean = np.mean(power_spectrum, axis=1)
    
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Only apply frequency filtering if fmin and fmax are specified
    if fmin is not None and fmax is not None:
        # Create mask for frequency range
        freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
        
        if not np.any(freq_mask):
            raise ValueError(f"No frequencies found in range {fmin}-{fmax} Hz")
        
        # Only return the values within the specified frequency range
        filtered_frequencies = frequencies[freq_mask]
        filtered_psd = psd_mean[freq_mask]
        return filtered_frequencies, filtered_psd
    else:
        # Return full range
        return frequencies, psd_mean

def find_artifacts_dir():
    """
    Search upward from the current directory for a directory named 'code',
    then return its 'artifacts' subdirectory.
    Flask version: creates artifacts directly if Flask detected.
    """

    # Current working directory
    current = Path.cwd()
    
    # Detect Flask environment (check for Flask-specific files/directories)
    is_flask_env = (
        (current / 'temp_uploads').exists() or 
        (current / 'simple_app.py').exists() or
        os.environ.get('FLASK_APP') is not None
    )
    
    if is_flask_env:
        # Flask version: create artifacts directory directly
        artifacts_dir = current / 'artifacts'
        if artifacts_dir.exists():
            print(f"Flask mode: Found existing artifacts directory: {artifacts_dir}")
            return artifacts_dir
        else:
            artifacts_dir.mkdir(exist_ok=True)
            print(f"Flask mode: Created artifacts directory: {artifacts_dir}")
            return artifacts_dir    

    # Check subdirectories of current directory first
    code_dir = current / 'code'
    if code_dir.exists():
        artifacts_dir = code_dir / 'artifacts'
        if artifacts_dir.exists():
            print(f"Found existing artifacts directory: {artifacts_dir}")
            return artifacts_dir
        else:
            artifacts_dir.mkdir(exist_ok=True)
            print(f"Created artifacts directory: {artifacts_dir}")
            return artifacts_dir

    # Search upwards for supdirectories named code - original logic
    for parent in [current] + list(current.parents):
        if parent.name == 'code':
            artifacts_dir = parent / 'artifacts'
            if artifacts_dir.exists():
                print(f"Found existing artifacts directory: {artifacts_dir}")
                return artifacts_dir
            else:
                artifacts_dir.mkdir(exist_ok=True)
                print(f"Created artifacts directory: {artifacts_dir}")
                return artifacts_dir
                #raise FileNotFoundError(f"'artifacts' directory does not exist in {parent}")
    
    raise FileNotFoundError("Could not find 'code' directory in any parent path.")








