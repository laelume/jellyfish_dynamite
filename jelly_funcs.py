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
    CHANGED from 'code/artifacts' directory reliance!!
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




# Dual-Resolution PSD and Spectrogram Calculation, 
# Interpolates spectrogram to smooth and restore info and, most importantly, match x-axis scaling
def calculate_psd_spectro(audio_path, 
                            # PSD Parameters (frequency-optimized)
                            psd_n_fft=2048,
                            psd_hop_length=None,
                            
                            # Spectrogram Parameters (time-optimized)  
                            spec_n_fft=1024,
                            spec_hop_length=None,
                            
                            # Control Flags
                            use_dual_resolution=True,
                            verbose=False):
    """Calculate both PSD and spectrogram with dual resolution - PSD optimized for frequency, spectrogram for time."""
    
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

    # Set hop lengths - be careful about x-axis scaling mismatches!!!
    if psd_hop_length is None:
        psd_hop_length = psd_n_fft // 16
    if spec_hop_length is None:
        spec_hop_length = spec_n_fft // 16

    if verbose:
        print(f"PSD: {psd_n_fft}-point FFT, {psd_hop_length} hop")
        print(f"Spectrogram: {spec_n_fft}-point FFT, {spec_hop_length} hop")
        print(f"Dual resolution: {'ON' if use_dual_resolution else 'OFF'}")



    # DUAL RESOLUTION LOGIC WITH INTERPOLATION
    if use_dual_resolution:
        # HIGH FREQUENCY RESOLUTION PSD 
        stft_psd = librosa.stft(y, n_fft=psd_n_fft, hop_length=psd_hop_length)
        power_spectrum_psd = np.abs(stft_psd)**2
        psd_mean = np.mean(power_spectrum_psd, axis=1)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=psd_n_fft)  # Master frequency grid
        
        # HIGH TIME RESOLUTION SPECTROGRAM
        stft_spec = librosa.stft(y, n_fft=spec_n_fft, hop_length=spec_hop_length)
        power_spectrum_spec = np.abs(stft_spec)**2
        times = librosa.frames_to_time(np.arange(power_spectrum_spec.shape[1]), 
                                       sr=sr, hop_length=spec_hop_length)
        spec_frequencies = librosa.fft_frequencies(sr=sr, n_fft=spec_n_fft)
        
        # INTERPOLATE SPECTROGRAM TO MATCH PSD FREQUENCY GRID
        from scipy.interpolate import interp1d
        interpolated_spectrogram = np.zeros((len(frequencies), power_spectrum_spec.shape[1]))
        
        if verbose:
            print(f"Interpolating spectrogram from {len(spec_frequencies)} to {len(frequencies)} freq bins")
        
        # Interpolate each time slice of the spectrogram
        for time_idx in range(power_spectrum_spec.shape[1]):
            # Create interpolation function for this time slice
            interp_func = interp1d(
                spec_frequencies, 
                power_spectrum_spec[:, time_idx], 
                kind='linear',           # Linear interpolation
                bounds_error=False,     # Don't error if outside bounds
                fill_value=0           # Use 0 for frequencies outside range
            )
            # Apply interpolation to get values at PSD frequency grid
            interpolated_spectrogram[:, time_idx] = interp_func(frequencies)
        
        power_spectrum = interpolated_spectrogram
        
        if verbose:
            print(f"Original spectrogram shape: {power_spectrum_spec.shape}")
            print(f"Interpolated spectrogram shape: {power_spectrum.shape}")
            print(f"PSD frequency resolution: {frequencies[1] - frequencies[0]:.2f} Hz/bin")
            print(f"Original spec resolution: {spec_frequencies[1] - spec_frequencies[0]:.2f} Hz/bin")
        

    else:
        # SINGLE RESOLUTION (original behavior)
        hop_length = psd_n_fft // 16
        stft_result = librosa.stft(y, n_fft=psd_n_fft, hop_length=hop_length)
        power_spectrum = np.abs(stft_result)**2
        
        # Get time axis
        times = librosa.frames_to_time(np.arange(power_spectrum.shape[1]), 
                                       sr=sr, hop_length=hop_length)
        
        # PSD is time-averaged
        psd_mean = np.mean(power_spectrum, axis=1)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=psd_n_fft)

    # Return full range, no filtering
    return frequencies, times, power_spectrum, psd_mean


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
        (current / 'jelly_app.py').exists() or
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
    
    # Fallback: create artifacts in current directory if no 'code' dir found
    artifacts_dir = current / 'artifacts'
    if not artifacts_dir.exists():
        artifacts_dir.mkdir(exist_ok=True)
        print(f"Created artifacts directory in current folder: {artifacts_dir}")
    else:
        print(f"Using existing artifacts directory: {artifacts_dir}")
    return artifacts_dir
