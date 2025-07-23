# jellyfish_plotly_browser.py

# Consolidated Interactive PSD Analysis Tool
# From jellyfish_super_scratch, using jellyfish_dynamite html
# Combines multiple spectral analysis methods with interactive visualization
# Uses jellyfush_dynamite html as the format for the browser-based plotting

from jellyfish_dynamite import save_jellyfish_template

from jinja2 import Template
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted
from scipy.signal import find_peaks
import matplotlib
from matplotlib.widgets import Button
import json
import librosa
from pathlib import Path
import warnings
import pywt

import scipy.signal
from scipy import signal

from scipy import interpolate

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import networkx as nx
import time
from datetime import datetime
import webbrowser

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
from bokeh.io import output_file, save, show
from bokeh.embed import file_html
from bokeh.resources import CDN
import json

from pathlib import Path
import importlib
import sys

#import all_functions_psd_rebuilt as alllpsd
#importlib.reload(alllpsd)

import jelly_funcs as jelfun
importlib.reload(jelfun)

# 2453 [58] has fewest number of chirps. 70 directories total. 
slicedir = Path('tranche/slices')
all_slicedirs=jelfun.get_subdir_pathlist(slicedir) 

mkdd = jelfun.make_daily_directory
daily_dir = mkdd()
daily_dir

thisfile='jellyfish_super_scratch'
parent_dir=os.path.join(daily_dir, thisfile)
os.makedirs(parent_dir, exist_ok=True)
parent_dir

slicedir = Path('tranche/slices')
all_slicedirs=jelfun.get_subdir_pathlist(slicedir) # 58: 2453 has fewest number of chirps. 70 directories total. 

warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal of length=.*")


# ==================== PSD CALCULATION METHODS ====================

def cqt_based_psd(audio_path, bins_per_octave=36, n_bins=144, fmin=20.0, fmax=None, hop_length=512, n_fft=2048):
    """Calculate PSD using Constant-Q Transform."""
    y, sr = librosa.load(audio_path, sr=None)
    
    if fmax is None:
        fmax = sr / 2
    if hop_length is None:
        hop_length = n_fft // 4
    
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, 
                    bins_per_octave=bins_per_octave, hop_length=hop_length)
    power_spectrum = np.abs(C)**2
    psd_mean = np.mean(power_spectrum, axis=1)
    frequencies = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    
    return frequencies, psd_mean

def multi_resolution_psd(audio_path, fft_sizes=[512, 1024, 2048, 4096], n_fft=None, hop_length=None):
    """Calculate PSD using multiple FFT window sizes."""
    y, sr = librosa.load(audio_path, sr=None)
    fft_sizes = sorted(fft_sizes)
    
    if n_fft is None:
        n_fft = max(fft_sizes)
    
    cutoff_freqs = []
    for i in range(len(fft_sizes) - 1):
        cutoff = np.sqrt(sr / fft_sizes[i] * sr / fft_sizes[i+1])
        cutoff_freqs.append(cutoff)
    cutoff_freqs.append(sr/2)
    
    out_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    out_psd = np.zeros_like(out_frequencies)
    
    for i, curr_n_fft in enumerate(fft_sizes):
        curr_hop_length = hop_length if hop_length else curr_n_fft // 4
        
        stft_result = librosa.stft(y, n_fft=curr_n_fft, hop_length=curr_hop_length)
        power_spectrum = np.abs(stft_result)**2
        curr_psd = np.mean(power_spectrum, axis=1)
        curr_freqs = librosa.fft_frequencies(sr=sr, n_fft=curr_n_fft)
        
        if i == 0:
            freq_mask = (out_frequencies > cutoff_freqs[i])
        elif i == len(fft_sizes) - 1:
            freq_mask = (out_frequencies <= cutoff_freqs[i-1])
        else:
            freq_mask = (out_frequencies > cutoff_freqs[i-1]) & (out_frequencies <= cutoff_freqs[i])
        
        f_interp = interp1d(curr_freqs, curr_psd, kind='linear', 
                           bounds_error=False, fill_value=(curr_psd[0], curr_psd[-1]))
        out_psd[freq_mask] = f_interp(out_frequencies[freq_mask])
        
        if i == len(fft_sizes) - 1:
            zero_mask = (out_psd == 0)
            if np.any(zero_mask):
                out_psd[zero_mask] = f_interp(out_frequencies[zero_mask])
    
    return out_frequencies, out_psd

def chirplet_transform(audio_path, n_chirps=100, min_freq=20, max_freq=5000, n_fft=None):
    """Simplified chirplet transform for adaptive time-frequency analysis."""
    
    y, sr = librosa.load(audio_path, sr=None)
    
    frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), n_chirps)
    chirp_energies = np.zeros(n_chirps)
    
    for i, freq in enumerate(frequencies):
        t = np.arange(0, len(y)/sr, 1/sr)
        chirp = signal.chirp(t, f0=freq*0.8, f1=freq*1.2, t1=len(y)/sr, method='logarithmic')
        window = signal.windows.hann(len(chirp))
        chirp *= window
        correlation = signal.correlate(y, chirp, mode='same')
        chirp_energies[i] = np.max(np.abs(correlation)**2)
    
    return frequencies, chirp_energies

def chirplet_transform_zero_padding(audio_path, n_chirps=100, min_freq=20, max_freq=5000, n_fft=2048):
    """Simplified chirplet transform with zero-padding."""
    
    y, sr = librosa.load(audio_path, sr=None)
    
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), 'constant')
    
    frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), n_chirps)
    chirp_energies = np.zeros(n_chirps)
    
    for i, freq in enumerate(frequencies):
        t = np.arange(0, len(y)/sr, 1/sr)
        chirp = signal.chirp(t, f0=freq*0.8, f1=freq*1.2, t1=len(y)/sr, method='logarithmic')
        window = signal.windows.hann(len(chirp))
        chirp *= window
        correlation = signal.correlate(y, chirp, mode='same')
        chirp_energies[i] = np.max(np.abs(correlation)**2)
    
    return frequencies, chirp_energies

def wavelet_packet_psd(audio_path, wavelet='sym8', max_level=8, hop_length=None, n_fft=2048):
    """Calculate PSD using Wavelet Packet Decomposition."""
    
    y, sr = librosa.load(audio_path, sr=None)
    print(f"Original Wavelet - Signal length: {len(y)}, Sample rate: {sr}")
    
    # Calculate padded length for better frequency resolution
    pad_length = 2**int(np.ceil(np.log2(len(y))))
    y_padded = np.pad(y, (0, pad_length - len(y)), 'constant')
    
    # Calculate optimal level based on the amount of data and desired frequency resolution
    optimal_level = min(int(np.log2(n_fft)), int(np.log2(len(y_padded))) - 2)
    
    if max_level > optimal_level:
        print(f"WARNING: Adjusting max_level from {max_level} to {optimal_level} for better accuracy")
        max_level = optimal_level
    
    if hop_length is None:
        hop_length = n_fft // 4
    
    segment_size = n_fft
    
    if len(y) < segment_size:
        y = np.pad(y, (0, segment_size - len(y)), 'constant')
    
    # Remove the try-except that was silently failing
    wp = pywt.WaveletPacket(data=y, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    nodes = [node for node in wp.get_level(max_level, 'natural')]
    powers = [np.mean(np.abs(node.data)**2) for node in nodes]
    powers = [max(p, 1e-10) for p in powers]
    
    # Define correction factors to fix frequency shift issue
    correction_factors = {
        'db4': 1.15,
        'db8': 1.12,
        'sym8': 1.05,
        'coif5': 1.03
    }
    correction = correction_factors.get(wavelet, 1.0)
    
    freqs = []
    nyquist = sr / 2
    bands = 2**max_level
    for i in range(bands):
        low = i * nyquist / bands * correction
        high = (i + 1) * nyquist / bands * correction
        center = (low + high) / 2
        freqs.append(center)
    
    if sum(powers) > 0:
        powers = [p / sum(powers) for p in powers]
    else:
        raise ValueError(f"All wavelet powers are zero for file {audio_path}. This indicates a problem with the input signal or wavelet decomposition.")
        
    print(f"Original Wavelet - Success. PSD range: {min(powers):.2e} to {max(powers):.2e}")
    return np.array(freqs), np.array(powers)

def improved_wavelet_packet_psd(audio_path, wavelet='sym8', max_level=8, hop_length=None, n_fft=2048):
    """Improved version of wavelet packet PSD with better frequency resolution."""
    
    y, sr = librosa.load(audio_path, sr=None)
    print(f"Improved Wavelet - Signal length: {len(y)}, Sample rate: {sr}")
    
    pow2_length = 2**int(np.ceil(np.log2(len(y))))
    if len(y) < pow2_length:
        y = np.pad(y, (0, pow2_length - len(y)), 'constant')
    
    optimal_level = min(int(np.log2(n_fft)), int(np.log2(len(y))) - 2)
    if max_level > optimal_level:
        print(f"WARNING: Adjusting max_level from {max_level} to {optimal_level} for better accuracy")
        max_level = optimal_level
    
    out_freqs = np.linspace(20, sr/2, n_fft//2)
    out_psd = np.zeros_like(out_freqs)
    
    wp = pywt.WaveletPacket(data=y, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    nodes = [node for node in wp.get_level(max_level, 'natural')]
    
    if not nodes:
        raise ValueError(f"No nodes found in wavelet packet decomposition for file {audio_path}. Check input signal and parameters.")
    
    energies = []
    for node in nodes:
        energy = np.sum(np.abs(node.data)**2)
        energies.append(max(energy, 1e-20))
    
    total_energy = sum(energies)
    if total_energy == 0:
        raise ValueError(f"Total energy is zero for file {audio_path}. This indicates a problem with the input signal.")
    
    # Calculate actual frequency bands without any "correction"
    bands = 2**max_level
    band_width = sr / (2 * bands)
    band_centers = np.array([(i * band_width + band_width/2) for i in range(bands)])
    powers = np.array(energies) / total_energy
    
    if len(band_centers) > 1:
        interp_func = interpolate.interp1d(
            band_centers, powers, 
            kind='linear',
            bounds_error=False, 
            fill_value=(powers[0], powers[-1])
        )
        out_psd = interp_func(out_freqs)
    else:
        out_psd = np.ones_like(out_freqs) * powers[0]
    
    window_length = min(21, len(out_psd) // 5 * 2 + 1)
    if window_length > 3 and window_length % 2 == 1:
        out_psd = savgol_filter(out_psd, window_length, 2)
    
    out_psd = np.maximum(out_psd, 1e-10)
    
    print(f"Improved Wavelet - Success. PSD range: {np.min(out_psd):.2e} to {np.max(out_psd):.2e}")
    return out_freqs, out_psd

def stationary_wavelet_psd(audio_path, wavelet='sym8', max_level=6, n_fft=2048):
    """Calculate PSD using Stationary Wavelet Transform (shift-invariant)."""
    
    y, sr = librosa.load(audio_path, sr=None)
    print(f"Stationary Wavelet - Signal length: {len(y)}, Sample rate: {sr}")
    
    target_length = int(2**np.ceil(np.log2(len(y))))
    y_padded = np.pad(y, (0, target_length - len(y)), 'constant')
    
    optimal_level = min(int(np.log2(n_fft)), int(np.log2(len(y_padded))) - 3)
    
    if max_level > optimal_level:
        print(f"WARNING: Adjusting max_level from {max_level} to {optimal_level}")
        max_level = max(1, optimal_level)
    
    out_freqs = np.linspace(20, sr/2, n_fft//2)
    out_psd = np.zeros_like(out_freqs)
    
    # Check if level is too high for the signal length
    if max_level >= int(np.log2(len(y_padded))):
        raise ValueError(f"max_level {max_level} is too high for signal length {len(y_padded)}. "
                        f"Maximum possible level is {int(np.log2(len(y_padded))) - 1}")
    
    coeffs = pywt.swt(y_padded, wavelet, level=max_level)
    
    level_powers = []
    level_freqs = []
    
    for i, (cA, cD) in enumerate(coeffs):
        level = max_level - i
        power = np.mean(np.abs(cD)**2)
        
        # Calculate actual frequency bands without any "correction"
        band_width = sr / (2**(level+1))
        low_freq = band_width
        high_freq = 2 * band_width
        center_freq = (low_freq + high_freq) / 2
        
        level_powers.append(power)
        level_freqs.append(center_freq)
    
    power = np.mean(np.abs(coeffs[-1][0])**2)
    center_freq = sr / (2**(max_level+1)) / 2
    level_powers.append(power)
    level_freqs.append(center_freq)
    
    total_power = sum(level_powers)
    if total_power > 0:
        level_powers = [p / total_power for p in level_powers]
    else:
        raise ValueError(f"Total power is zero for file {audio_path} using stationary wavelet transform. Check input signal.")
    
    for i, (freq, power) in enumerate(zip(level_freqs, level_powers)):
        idx = np.abs(out_freqs - freq).argmin()
        out_psd[idx] = power
    
    valid_indices = np.where(out_psd > 0)[0]
    if len(valid_indices) > 1:
        f_interp = interp1d(
            out_freqs[valid_indices], 
            out_psd[valid_indices],
            kind='linear', 
            bounds_error=False, 
            fill_value=(out_psd[valid_indices[0]], out_psd[valid_indices[-1]])
        )
        out_psd = f_interp(out_freqs)
    
    window_length = min(21, len(out_psd) // 5 * 2 + 1)
    if window_length > 3:
        out_psd = savgol_filter(out_psd, window_length, 2)
    
    out_psd = np.maximum(out_psd, 1e-10)
    
    print(f"Stationary Wavelet - Success. PSD range: {np.min(out_psd):.2e} to {np.max(out_psd):.2e}")
    return out_freqs, out_psd

# ==================== INTERACTIVE PLOT CLASS ====================

class EnhancedInteractiveHarmonicPlot:
    def __init__(self, frequencies, psd, filename, ax, max_peaks=40, max_pairs=10, 
                 is_db_scale=True, peak_fmin=None, peak_fmax=None, 
                 plot_fmin=None, plot_fmax=None, 
                 height_percentile=0.5, prominence_factor=0.04,
                 min_width=0.5, method_name="FFT", top_padding_db=10):
        
        self.frequencies = frequencies
        self.psd = psd.copy()
        self.filename = filename
        self.ax = ax
        self.max_peaks = max_peaks
        self.max_pairs = max_pairs
        self.selected_peaks = []
        self.pairs = []
        self.colors = ['red', 'green', 'purple', 'orange', 'brown']
        self.click_tolerance = 150
        self.is_db_scale = is_db_scale
        self.height_percentile = height_percentile
        self.prominence_factor = prominence_factor
        self.min_width = min_width
        self.method_name = method_name
        self.last_click_time = 0
        self.last_click_button = None
        
        ''' 
        # Legacy Conversion Stuff
        # # Convert to dB scale if needed
        # if is_db_scale:
        #     psd_for_db = np.maximum(self.psd, 1e-10)
        #     self.psd_db = 10 * np.log10(psd_for_db)
        # else:
        #     self.psd_db = self.psd
        # self.original_psd = psd.copy()
        # # New scale conversion psd calculation methods: 
        # self.psd_linear = np.maximum(psd.copy(), 1e-15)  # Store linear version
        # self.psd_db = 10 * np.log10(self.psd_linear)     # Calculate dB version
        # #self.is_linear_scale = not is_db_scale           # Track current scale
        # self.is_db_scale = is_db_scale                   # Track current scale
        # self.current_psd = self.psd_db if is_db_scale else self.psd_linear
        # self.original_psd = self.psd_linear.copy()
        '''

        # FIXED: Properly initialize both linear and dB scales
        # Always store the original linear version
        self.psd_linear = np.maximum(psd.copy(), 1e-15)  # Store linear version
        self.original_psd = self.psd_linear.copy()       # Keep reference to original
        
        # Calculate dB version from linear
        self.psd_db = 10 * np.log10(self.psd_linear)
        
        # Set current PSD based on scale preference
        if is_db_scale:
            self.current_psd = self.psd_db
            self.psd = self.psd_db.copy()  # For backward compatibility
        else:
            self.current_psd = self.psd_linear
            self.psd = self.psd_linear.copy()  # For backward compatibility

        # Set frequency ranges
        self.peak_fmin = peak_fmin if peak_fmin is not None else np.min(frequencies)
        self.peak_fmax = peak_fmax if peak_fmax is not None else np.max(frequencies)
        self.plot_fmin = plot_fmin if plot_fmin is not None else np.min(frequencies)
        self.plot_fmax = plot_fmax if plot_fmax is not None else min(5000, np.max(frequencies))
        
        # Plot the PSD
        self.line, = ax.plot(frequencies, self.current_psd, 'k-', alpha=0.7)
        
        # Y-LIMITS
        # Calculate and store y-limits for both scales
        self._calculate_ylimits(top_padding_db)
        # Set initial y-limits based on current scale
        if self.is_db_scale:
            ax.set_ylim(self.ylim_db)
        else:
            ax.set_ylim(self.ylim_linear)
        
        '''
        # ax.autoscale(axis='y', tight=True)
        # y_min, y_max = ax.get_ylim()
        # ax.set_ylim(y_min, y_max + top_padding_db)
        # Padding moved to _calculate_ylimits()
        # Padding adjustment for linear vs. db scales
        # if self.is_db_scale:
        #     ax.set_ylim(y_min, y_max + top_padding_db)
        # else:
        #     linear_padding = (y_max - y_min) * 0.1  # 10% padding for linear
        #     ax.set_ylim(y_min, y_max + linear_padding)
        '''

        # X-LIMITS
        ax.set_xlim(self.plot_fmin, self.plot_fmax)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB)' if is_db_scale else 'PSD (linear)')
        title_scale = "(dB)" if is_db_scale else "(linear)"
        ax.set_title(f"{filename} [{self.peak_fmin}-{self.peak_fmax} Hz] >> {self.method_name} {title_scale}")
        
        # Find peaks using the current scale
        peak_freq_mask = (frequencies >= self.peak_fmin) & (frequencies <= self.peak_fmax)
        peak_detection_freqs = frequencies[peak_freq_mask]

        # Use dB scale for peak detection for consistency
        peak_detection_psd = self.psd_db[peak_freq_mask]
        
        # Adaptive peak detection
        min_desired_peaks = min(10, self.max_peaks)
        current_height_percentile = self.height_percentile
        current_prominence_factor = self.prominence_factor
        
        all_peaks_indices = []
        peak_properties = {}
        
        while len(all_peaks_indices) < min_desired_peaks and current_height_percentile > 0.2:
            all_peaks_indices, peak_properties = find_peaks(
                peak_detection_psd, 
                height=np.percentile(peak_detection_psd, current_height_percentile*100),
                prominence=current_prominence_factor*np.std(peak_detection_psd),
                width=max(self.min_width, 1.0)
            )
            
            if len(all_peaks_indices) < min_desired_peaks:
                current_height_percentile -= 0.05
                current_prominence_factor /= 1.5
        
        # Calculate peak widths
        self.freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1
        if 'widths' in peak_properties and len(peak_properties['widths']) > 0:
            self.peak_widths = peak_properties['widths'] * self.freq_resolution
        else:
            self.peak_widths = np.zeros(len(all_peaks_indices))
        
        # Map to original frequency indices
        peak_indices_in_original = np.where(peak_freq_mask)[0][all_peaks_indices]
        
        # Limit to max_peaks
        if len(peak_indices_in_original) > self.max_peaks:
            peak_prominences = np.array([self.psd_db[i] for i in peak_indices_in_original])
            top_indices = np.argsort(peak_prominences)[::-1][:self.max_peaks]
            peak_indices_in_original = peak_indices_in_original[top_indices]
            if len(self.peak_widths) == len(all_peaks_indices):
                self.peak_widths = self.peak_widths[top_indices]
        

        self.peak_freqs = frequencies[peak_indices_in_original]
        self.peak_powers = self.current_psd[peak_indices_in_original]


        # Plot peaks - store reference for updates
        self.grey_peaks_line, = ax.plot(self.peak_freqs, self.peak_powers, 'o', color='gray', markersize=4, alpha=0.5, picker=5)

        # Add vertical lines for peaks
        self.all_peak_lines = []
        for freq in self.peak_freqs:
            vline = ax.axvline(x=freq, color='pink', linestyle='--', 
                            alpha=0.6, linewidth=0.5)
            self.all_peak_lines.append(vline)
        
        # Initialize interaction variables
        self.fig = ax.figure
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.peak_labels = []
        self.selected_markers = []
        self.pair_lines = []
        self.pair_markers = []
        self.width_markers = []
        self.legend_text = []
        self.graph = nx.Graph()
        
        self.fig.canvas.draw_idle()


    def _calculate_ylimits(self, top_padding_db=10):
        """Calculate and store y-limits for both dB and linear scales"""
        # Calculate limits for dB scale
        db_min, db_max = np.min(self.psd_db), np.max(self.psd_db)
        self.ylim_db = (db_min, db_max + top_padding_db)
        
        # Calculate limits for linear scale
        linear_min, linear_max = np.min(self.psd_linear), np.max(self.psd_linear)
        linear_padding = (linear_max - linear_min) * 0.1  # 10% padding
        self.ylim_linear = (linear_min, linear_max + linear_padding)


    def toggle_scale(self):
        """Toggle between dB and linear scale - with peak updates"""
        self.is_db_scale = not self.is_db_scale
        self.current_psd = self.psd_db if self.is_db_scale else self.psd_linear
        
        # Update the main line
        self.line.set_ydata(self.current_psd)

        # Update y-axis limits to appropriate scale
        if self.is_db_scale:
            self.ax.set_ylim(self.ylim_db)
        else:
            self.ax.set_ylim(self.ylim_linear)

        # Update labels
        ylabel = 'PSD (dB)' if self.is_db_scale else 'PSD (linear)'
        self.ax.set_ylabel(ylabel)
        
        scale_text = "(dB)" if self.is_db_scale else "(linear)"
        title = self.ax.get_title()
        if "(dB)" in title or "(linear)" in title:
            title = title.replace("(dB)", scale_text).replace("(linear)", scale_text)
        self.ax.set_title(title)

        # Update peak powers for current scale
        if hasattr(self, 'peak_freqs') and len(self.peak_freqs) > 0:
            peak_indices = [np.argmin(np.abs(self.frequencies - f)) for f in self.peak_freqs]
            self.peak_powers = self.current_psd[peak_indices]

        # Find and update the grey peak marker dots
        if hasattr(self, 'grey_peaks_line'):
            self.grey_peaks_line.set_ydata(self.peak_powers)

        # Update all interactive elements
        self.update_display()  # This redraws all selected peaks, pairs, and lines
        self.fig.canvas.draw_idle()
        
        print(f"Toggled to {'dB' if self.is_db_scale else 'linear'} scale")


    def get_current_peak_power(self, freq):
        """Get peak power in current scale"""
        idx = np.argmin(np.abs(self.peak_freqs - freq))
        return self.peak_powers[idx]

    def find_nearest_peak(self, x, y):
        """Find the nearest peak to the clicked position."""
        if len(self.peak_freqs) == 0:
            return None
        
        distances = np.abs(self.peak_freqs - x)
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] < self.click_tolerance:
            return nearest_idx
        return None

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
        
        current_time = time.time()
        is_double_click = False
        is_triple_click = False

        if hasattr(self, 'last_click_time') and hasattr(self, 'last_click_button'):
            time_diff = current_time - self.last_click_time
            
            if hasattr(self, 'double_click_time') and time_diff < 0.3 and event.button == self.last_click_button:
                time_since_double = current_time - self.double_click_time
                if time_since_double < 0.5:
                    is_triple_click = True
                    print("Triple click detected!")
            elif time_diff < 0.3 and event.button == self.last_click_button:
                is_double_click = True
                self.double_click_time = current_time

        self.last_click_time = current_time
        self.last_click_button = event.button

        # Handle left click events
        if event.button == 1:  # Left click
            if is_triple_click:  # Remove ALL nodes and pairs
                node_count = len(self.graph.nodes())
                pair_count = len(self.pairs)
                
                self.graph.clear()
                self.pairs = []
                self.selected_peaks = []
                
                for _, label in self.peak_labels:
                    if hasattr(label, 'remove'):
                        label.remove()
                self.peak_labels = []
                
                print(f"Triple-click: Removed all {node_count} nodes and {pair_count} pairs")
                
            elif is_double_click:  # Remove point and any pairs it's part of
                peak_idx = self.find_nearest_peak(event.xdata, event.ydata)
                if peak_idx is not None:
                    freq = self.peak_freqs[peak_idx]
                    power = self.peak_powers[peak_idx]
                    
                    pairs_to_remove = []
                    for pair_idx, pair in enumerate(self.pairs):
                        if abs(pair['f0'] - freq) < self.click_tolerance or abs(pair['f1'] - freq) < self.click_tolerance:
                            pairs_to_remove.append(pair_idx)
                    
                    if self.graph.has_node(freq):
                        connected_nodes = list(self.graph.neighbors(freq))
                        self.graph.remove_node(freq)
                        print(f"Removed node {freq:.1f} Hz with {len(connected_nodes)} connections")
                    
                    for pair_idx in sorted(pairs_to_remove, reverse=True):
                        other_freq = self.pairs[pair_idx]['f1'] if abs(self.pairs[pair_idx]['f0'] - freq) < self.click_tolerance else self.pairs[pair_idx]['f0']
                        if other_freq not in self.selected_peaks:
                            self.selected_peaks.append(other_freq)
                        self.pairs.pop(pair_idx)
                        print(f"Removed pair containing {freq:.1f} Hz, kept other peak")
                    
                    if freq in self.selected_peaks:
                        self.selected_peaks.remove(freq)
                        self.remove_peak_label(freq)
                        print(f"Removed selected peak at {freq:.1f} Hz")
                    
            else:  # Single left click - add point
                peak_idx = self.find_nearest_peak(event.xdata, event.ydata)
                if peak_idx is not None:
                    freq = self.peak_freqs[peak_idx]
                    power = self.peak_powers[peak_idx]
                    
                    if freq not in self.selected_peaks:
                        self.selected_peaks.append(freq)
                        
                        if not self.graph.has_node(freq):
                            self.graph.add_node(freq, power=power)
                            print(f"Added peak {freq:.1f} Hz to graph")
                        
                        marker, = self.ax.plot(freq, power, 'o', color='blue', 
                                            markersize=8, markeredgecolor='black')
                        self.selected_markers.append(marker)
                        
                        self.add_peak_label(freq, power)
                        print(f"Added peak at {freq:.1f} Hz")

        # Handle right click events
        elif event.button == 3:  # Right click
            if is_triple_click:  # Remove ALL edges but keep nodes
                edge_count = len(self.graph.edges())
                pair_count = len(self.pairs)
                
                nodes = list(self.graph.nodes(data=True))
                
                for edge in list(self.graph.edges()):
                    self.graph.remove_edge(*edge)
                
                self.pairs = []
                
                for node, _ in nodes:
                    if node not in self.selected_peaks:
                        self.selected_peaks.append(node)
                
                print(f"Triple-click: Removed all {edge_count} edges and {pair_count} pairs, kept {len(nodes)} nodes")
                
            elif is_double_click:  # Remove pair
                peak_idx = self.find_nearest_peak(event.xdata, event.ydata)
                if peak_idx is not None:
                    freq = self.peak_freqs[peak_idx]
                    
                    for pair_idx, pair in enumerate(self.pairs):
                        if abs(pair['f0'] - freq) < self.click_tolerance or abs(pair['f1'] - freq) < self.click_tolerance:
                            f0, f1 = pair['f0'], pair['f1']
                            
                            self.pairs.pop(pair_idx)
                            
                            if f0 not in self.selected_peaks:
                                self.selected_peaks.append(f0)
                            if f1 not in self.selected_peaks:
                                self.selected_peaks.append(f1)
                            
                            if self.graph.has_edge(f0, f1):
                                self.graph.remove_edge(f0, f1)
                                print(f"Removed graph edge between {f0:.1f} Hz and {f1:.1f} Hz")
                            
                            print(f"Removed pair containing {freq:.1f} Hz, kept peaks")
                            break
                            
            else:  # Single right click - create pair
                if len(self.selected_peaks) >= 2 and len(self.pairs) < self.max_pairs:
                    f1 = self.selected_peaks[-2]
                    f2 = self.selected_peaks[-1]
                    
                    f0, f1 = sorted([f1, f2])
                    
                    color = self.colors[len(self.pairs) % len(self.colors)]
                    self.pairs.append({'f0': f0, 'f1': f1, 'color': color})
                    
                    ratio = f1 / f0
                    self.graph.add_edge(f0, f1, color=color, ratio=ratio)
                    print(f"Created edge in graph: {f0:.1f} Hz <-> {f1:.1f} Hz (ratio: {ratio:.3f})")
                    
                    for peak in [f0, f1]:
                        if peak in self.selected_peaks:
                            self.selected_peaks.remove(peak)
                    
                    print(f"Created pair: {f0:.1f} Hz and {f1:.1f} Hz, ratio: {f1/f0:.3f}")
        
        self.update_display()

    def on_key(self, event):
        """Handle key press events."""
        if event.key == 'c':  # Clear current selections
            self.selected_peaks = []
        elif event.key == 'r':  # Reset everything
            self.selected_peaks = []
            self.pairs = []
            self.graph.clear()
            print("Reset all pairs, selections, and graph data")
        elif event.key == 'a':  # Create fully connected graph
            self._create_fully_connected_graph()
            print("Created fully connected graph of selected points")
        elif event.key == 'm':  # Matrix view
            self._show_matrix_debug()
        elif event.key == 'd':  # Toggle db scale
            self.toggle_scale()

        self.update_display()

    def _create_fully_connected_graph(self):
        """Create a fully connected graph from all selected peaks and existing nodes."""
        nodes_to_connect = set(self.graph.nodes())
        for freq in self.selected_peaks:
            nodes_to_connect.add(freq)
        
        nodes_list = sorted(list(nodes_to_connect))
        num_nodes = len(nodes_list)
        
        if num_nodes < 2:
            print("Need at least 2 points to create connections")
            return
        
        print(f"Creating fully connected graph with {num_nodes} nodes")
        
        edges_added = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                f0 = nodes_list[i]
                f1 = nodes_list[j]
                
                if self.graph.has_edge(f0, f1):
                    continue
                
                ratio = f1 / f0
                color_idx = edges_added % len(self.colors)
                color = self.colors[color_idx]
                
                self.graph.add_edge(f0, f1, color=color, ratio=ratio)
                
                if len(self.pairs) < self.max_pairs:
                    self.pairs.append({'f0': f0, 'f1': f1, 'color': color})
                
                edges_added += 1
        
        for freq in list(self.selected_peaks):
            for pair in self.pairs:
                if abs(pair['f0'] - freq) < self.click_tolerance or abs(pair['f1'] - freq) < self.click_tolerance:
                    if freq in self.selected_peaks:
                        self.selected_peaks.remove(freq)
                    break
        
        print(f"Added {edges_added} new connections to the graph")

    def _show_matrix_debug(self):
        """Show a debug representation of the relationship matrix."""
        if not self.graph.nodes():
            print("No nodes in graph - cannot create matrix")
            return
            
        nodes = sorted(self.graph.nodes())
        print(f"Graph contains {len(nodes)} nodes: {[f'{n:.1f}' for n in nodes]}")
        
        print("\nFrequency ratio matrix:")
        print("   " + "".join(f"{n:8.0f} " for n in nodes))
        print("   " + "--------" * len(nodes))
        
        for i, n1 in enumerate(nodes):
            row = f"{n1:4.0f}|"
            for j, n2 in enumerate(nodes):
                if i == j:
                    row += f"{1.0:8.3f} "
                elif self.graph.has_edge(n1, n2):
                    edge_data = self.graph.get_edge_data(n1, n2)
                    ratio = edge_data.get('ratio', n2/n1)
                    row += f"{ratio:8.3f} "
                else:
                    row += "        - "
            print(row)
        
        print("\nEdges:")
        for edge in self.graph.edges(data=True):
            n1, n2, data = edge
            print(f"Edge: {n1:.1f} Hz <-> {n2:.1f} Hz, ratio: {data.get('ratio', n2/n1):.3f}")

    # def add_peak_label(self, freq, power):
    #     """Add a label next to a peak showing its frequency."""
    #     label = self.ax.text(freq + -100, power + 4, f"{freq:.0f}", 
    #                     fontsize=10, color='blue', 
    #                     verticalalignment='center')
    #     self.peak_labels.append((freq, label))


    def add_peak_label(self, freq, power):
        """Add a label next to a peak showing its frequency."""
        if self.is_db_scale:
            offset = 4  # dB offset
        else:
            offset = power * 0.05  # 5% of power for linear
        
        label = self.ax.text(freq + -100, power + offset, f"{freq:.0f}", 
                        fontsize=10, color='blue', 
                        verticalalignment='center')
        self.peak_labels.append((freq, label))

    def add_width_marker(self, freq, power, width):
        """Add a horizontal line showing the width of a peak."""
        if width > 0:
            half_width = width / 2
            width_line, = self.ax.plot([freq - half_width, freq + half_width], 
                                [power - 3, power - 3],
                                '-', color='lightgray', linewidth=2.0)
            self.width_markers.append(width_line)

    def remove_peak_label(self, freq):
        """Remove the label for a specific frequency."""
        for i, (peak_freq, label) in enumerate(self.peak_labels):
            if abs(peak_freq - freq) < self.click_tolerance:
                label.remove()
                self.peak_labels.pop(i)
                return

    def get_graph_data(self):
        """Return a serializable representation of the graph data."""
        data = {"nodes": [], "edges": []}
        
        for node in self.graph.nodes(data=True):
            node_id, attrs = node
            data["nodes"].append({
                "id": float(node_id),
                "frequency": float(node_id),
                "power": attrs.get("power", 0)
            })
        
        for edge in self.graph.edges(data=True):
            n1, n2, attrs = edge
            data["edges"].append({
                "source": float(n1),
                "target": float(n2),
                "ratio": attrs.get("ratio", float(n2)/float(n1)),
                "color": attrs.get("color", "red")
            })
        
        return data

    def update_display(self):
        """Update the visual display of the plot."""
        # Clear old markers and lines
        for marker in self.selected_markers:
            if marker in self.ax.lines:
                marker.remove()
        for line in self.pair_lines:
            if hasattr(line, 'remove'):
                line.remove()
        for marker in self.pair_markers:
            if marker in self.ax.lines:
                marker.remove()
        for marker in self.width_markers:
            if hasattr(marker, 'remove'):
                marker.remove()

        # Clear vertical lines for all peaks
        if hasattr(self, 'all_peak_lines'):
            for line in self.all_peak_lines:
                if hasattr(line, 'remove'):
                    line.remove()
            self.all_peak_lines = []

        # Clear all peak labels
        for _, label in self.peak_labels:
            if hasattr(label, 'remove'):
                label.remove()
        self.peak_labels = []
        
        self.selected_markers = []
        self.pair_lines = []
        self.pair_markers = []
        self.width_markers = []

        # Draw vertical lines for ALL peaks
        for freq in self.peak_freqs:
            vline = self.ax.axvline(x=freq, color='pink', linestyle='--', 
                                alpha=0.6, linewidth=0.5, zorder=-3)
            if not hasattr(self, 'all_peak_lines'):
                self.all_peak_lines = []
            self.all_peak_lines.append(vline)





        # # Draw selected peaks
        # for freq in self.selected_peaks:
        #     idx = np.argmin(np.abs(self.peak_freqs - freq))
        #     power = self.peak_powers[idx]
        #     marker, = self.ax.plot(freq, power, 'o', color='blue', 
        #                         markersize=8, markeredgecolor='black')
        #     self.selected_markers.append(marker)
            
        #     self.add_peak_label(freq, power)
            
        #     if idx < len(self.peak_widths):
        #         self.add_width_marker(freq, power, self.peak_widths[idx])

        # Draw selected peaks (using current scale)
        for freq in self.selected_peaks:
            # Get power in current scale
            freq_idx = np.argmin(np.abs(self.frequencies - freq))
            if self.is_db_scale:
                power = self.psd_db[freq_idx]
            else:
                power = self.psd_linear[freq_idx]
            
            marker, = self.ax.plot(freq, power, 'o', color='blue', 
                                markersize=8, markeredgecolor='black')
            self.selected_markers.append(marker)
            
            self.add_peak_label(freq, power)



        # # Draw pairs
        # paired_freqs = set()
        
        # for pair in self.pairs:
        #     f0, f1 = pair['f0'], pair['f1']
        #     paired_freqs.add(f0)
        #     paired_freqs.add(f1)
        #     color = pair['color']
            
        #     f0_idx = np.argmin(np.abs(self.peak_freqs - f0))
        #     f1_idx = np.argmin(np.abs(self.peak_freqs - f1))
        #     f0_power = self.peak_powers[f0_idx]
        #     f1_power = self.peak_powers[f1_idx]
            
        #     # Draw markers
        #     m1, = self.ax.plot(f0, f0_power, 'o', color='blue', markersize=8, 
        #                     markeredgecolor='black')
        #     m2, = self.ax.plot(f1, f1_power, 'o', color='blue', markersize=8, 
        #                     markeredgecolor='black')
        #     self.pair_markers.extend([m1, m2])
            
        #     # Add labels for paired peaks
        #     self.add_peak_label(f0, f0_power)
        #     self.add_peak_label(f1, f1_power)
            
        #     # Add width markers for paired peaks
        #     if f0_idx < len(self.peak_widths):
        #         self.add_width_marker(f0, f0_power, self.peak_widths[f0_idx])
        #     if f1_idx < len(self.peak_widths):
        #         self.add_width_marker(f1, f1_power, self.peak_widths[f1_idx])
            
        #     # Draw connecting line
        #     line, = self.ax.plot([f0, f1], [f0_power, f1_power], '-', 
        #                     color=color, alpha=0.5, linewidth=1.3)
        #     self.pair_lines.append(line)
            
        #     # Draw vertical lines
        #     vline1 = self.ax.axvline(f0, color=color, linestyle='--', alpha=0.7)
        #     vline2 = self.ax.axvline(f1, color=color, linestyle='--', alpha=0.7)
        #     self.pair_lines.extend([vline1, vline2])

        # Draw pairs (using current scale)
        for pair in self.pairs:
            f0, f1 = pair['f0'], pair['f1']
            color = pair['color']
            
            # Get powers in current scale
            f0_idx = np.argmin(np.abs(self.frequencies - f0))
            f1_idx = np.argmin(np.abs(self.frequencies - f1))
            
            if self.is_db_scale:
                f0_power = self.psd_db[f0_idx]
                f1_power = self.psd_db[f1_idx]
            else:
                f0_power = self.psd_linear[f0_idx]
                f1_power = self.psd_linear[f1_idx]
            
            # Draw markers
            m1, = self.ax.plot(f0, f0_power, 'o', color='blue', markersize=8, 
                            markeredgecolor='black')
            m2, = self.ax.plot(f1, f1_power, 'o', color='blue', markersize=8, 
                            markeredgecolor='black')
            self.pair_markers.extend([m1, m2])
            
            # Add labels
            self.add_peak_label(f0, f0_power)
            self.add_peak_label(f1, f1_power)
            
            # Draw connecting line
            line, = self.ax.plot([f0, f1], [f0_power, f1_power], '-', 
                            color=color, alpha=0.5, linewidth=1.3)
            self.pair_lines.append(line)
            
            # Draw vertical lines
            vline1 = self.ax.axvline(f0, color=color, linestyle='--', alpha=0.7)
            vline2 = self.ax.axvline(f1, color=color, linestyle='--', alpha=0.7)
            self.pair_lines.extend([vline1, vline2])



        # Update legend
        self.update_legend()
        self.fig.canvas.draw_idle()

    def update_legend(self):
        """Update the legend display."""
        # Remove old legend
        for item in self.legend_text:
            if hasattr(item, 'remove'):
                item.remove()
        
        self.legend_text = []
        
        # Create legend text
        x_offset = 1.02
        y_start = 0.98
        line_height = 0.08
        
        # Add legend entries for each pair
        for i, pair in enumerate(self.pairs):
            f0, f1 = pair['f0'], pair['f1']
            ratio = f1 / f0
            color = pair['color']
            
            y_pos = y_start - (i * 3 * line_height)
            
            # f0 entry
            self.ax.plot([x_offset], [y_pos], 'o', color=color, 
                    transform=self.ax.transAxes, markersize=8, 
                    markeredgecolor='black', clip_on=False)
            text = self.ax.text(x_offset + 0.05, y_pos, f'{f0:.0f}', 
                            transform=self.ax.transAxes, verticalalignment='center',
                            fontsize=9)
            self.legend_text.append(text)
            
            # f1 entry
            self.ax.plot([x_offset], [y_pos - line_height], 's', color=color, 
                    transform=self.ax.transAxes, markersize=8, 
                    markeredgecolor='black', clip_on=False)
            text = self.ax.text(x_offset + 0.05, y_pos - line_height, f'{f1:.0f}', 
                            transform=self.ax.transAxes, verticalalignment='center',
                            fontsize=9)
            self.legend_text.append(text)
            
            # Ratio entry
            text = self.ax.text(x_offset + 0.05, y_pos - 2*line_height, f'ratio: {ratio:.2f}', 
                            transform=self.ax.transAxes, verticalalignment='center',
                            fontsize=9)
            self.legend_text.append(text)

# ==================== FILE SELECTION UTILITIES ====================

def select_audio_files(directory_path, file_patterns=None, file_indices=None, 
                      range_start=None, range_end=None, limit=None, 
                      extensions=('.wav',), verbose=True):
    """Select audio files from a directory with flexible filtering options."""
    all_files = natsorted([f for f in os.listdir(directory_path) 
                          if f.lower().endswith(extensions)])
    
    if verbose:
        print(f"Found {len(all_files)} audio files in directory")
    
    selected = all_files
    selection_desc = "all files"
    
    # Apply range selection if specified
    if range_start is not None:
        if range_end is None:
            range_end = len(all_files) - 1
        
        range_start = max(0, min(range_start, len(all_files) - 1))
        range_end = max(range_start, min(range_end, len(all_files) - 1))
        
        selected = all_files[range_start:range_end + 1]
        selection_desc = f"files {range_start} to {range_end}"
        
        if verbose:
            print(f"Range selection: {selection_desc} ({len(selected)} files)")
    
    # Apply specific indices if provided
    if file_indices is not None:
        valid_indices = [i for i in file_indices if 0 <= i < len(all_files)]
        
        if len(valid_indices) != len(file_indices) and verbose:
            print(f"Warning: {len(file_indices) - len(valid_indices)} indices were out of range")
        
        selected = [all_files[i] for i in valid_indices]
        selection_desc = f"files at indices {valid_indices}"
        
        if verbose:
            print(f"Index selection: {len(selected)} files at specified indices")
    
    # Apply pattern filtering
    if file_patterns is not None:
        filtered = []
        pattern_matches = {pattern: 0 for pattern in file_patterns}
        
        for filename in selected:
            for pattern in file_patterns:
                if pattern.lower() in filename.lower():
                    filtered.append(filename)
                    pattern_matches[pattern] += 1
                    break
        
        selected = filtered
        selection_desc = f"files matching patterns {file_patterns}"
        
        if verbose:
            print(f"Pattern filtering: {len(selected)} files matched")
            for pattern, count in pattern_matches.items():
                print(f"  - '{pattern}': {count} matches")
    
    # Apply limit
    if limit is not None and limit > 0:
        original_count = len(selected)
        selected = selected[:limit]
        selection_desc = f"first {limit} of {selection_desc}"
        
        if verbose and len(selected) < original_count:
            print(f"Limit applied: {len(selected)} of {original_count} files")
    
    if verbose:
        print(f"Final selection: {len(selected)} files")
        if len(selected) > 0:
            print(f"Sample of selected files:")
            for i, filename in enumerate(selected[:3]):
                print(f"  {i}. {filename}")
            if len(selected) > 3:
                print(f"  ... and {len(selected) - 3} more files")
    
    return selected

# ==================== MAIN ANALYSIS FUNCTIONS ====================

def compare_methods_psd_analysis(audio_directory, max_cols=4, max_pairs=5, 
                           n_fft=1024, peak_fmin=100, peak_fmax=6000,
                           plot_fmin=100, plot_fmax=6000,
                           height_percentile=0.6, prominence_factor=0.05,
                           min_width=0.6, methods=None, 
                           selected_files=None, use_db_scale=True):
    """
    Create an interactive PSD analysis for all audio files, using multiple methods.
    Each row displays a different audio file, and each column shows a different method.
    
    Args:
        audio_directory: Path to directory containing .wav files
        max_cols: Maximum number of columns in each row (methods)
        max_pairs: Maximum number of pairs per spectrogram
        n_fft: FFT window size for PSD calculation
        peak_fmin: Minimum frequency for peak detection (Hz)
        peak_fmax: Maximum frequency for peak detection (Hz)
        plot_fmin: Minimum frequency to display in plot (Hz)
        plot_fmax: Maximum frequency to display in plot (Hz)
        height_percentile: Peak detection height percentile threshold
        prominence_factor: Peak detection prominence factor
        min_width: Minimum width for peak detection
        methods: List of method names to use ["FFT", "CQT", "Chirplet Zero", "Multi-Res"]
        selected_files: List of filenames to process (if None, use all .wav files)
        use_db_scale: If True, display PSD in dB scale; if False, use linear scale
        
    Returns:
        Tuple of (figure, plots, save_function)
    """
    # Get audio files
    if selected_files is None:
        audio_files = natsorted([f for f in os.listdir(audio_directory) if f.endswith('.wav')])
    else: 
        audio_files = selected_files
    
    n_files = len(audio_files)
    
    if n_files == 0:
        print("No .wav files found.")
        return None, [], None
    
    # Set default plot range if not specified
    if plot_fmin is None:
        plot_fmin = peak_fmin
    
    # Define the methods to use
    if methods is None:
        methods = ["FFT", "Chirplet Zero"]  # Default to two common methods
    
    # Create method lookup dict - using alllpsd functions where available
    method_funcs = {
        "FFT": lambda path: jelfun.calculate_psd(
            path, n_fft=n_fft
        ),
        "CQT": lambda path: cqt_based_psd(
            path, bins_per_octave=36, n_bins=150, fmin=600.0, fmax=plot_fmax * 1.2, hop_length=n_fft//4, n_fft=n_fft
        ),
        "Wavelet": lambda path: wavelet_packet_psd(
            path, wavelet='sym8', max_level=6, hop_length=n_fft//4, n_fft=n_fft
        ),
        "Improved Wavelet": lambda path: improved_wavelet_packet_psd(
            path, wavelet='sym8', max_level=6, hop_length=n_fft//4, n_fft=n_fft
        ),
        "Stationary Wavelet": lambda path: stationary_wavelet_psd(
            path, wavelet='sym8', max_level=4, n_fft=n_fft
        ),
        "Chirplet": lambda path: chirplet_transform(
            path, n_chirps=100, min_freq=20, max_freq=plot_fmax * 1.2, n_fft=n_fft
        ),
        "Chirplet Zero": lambda path: chirplet_transform_zero_padding(
            path, n_chirps=100, min_freq=20, max_freq=plot_fmax * 1.2, n_fft=n_fft
        ),
        "Multi-Res": lambda path: multi_resolution_psd(
            path, fft_sizes=[512, 1024, 2048, 4096], n_fft=n_fft, hop_length=n_fft // 4
        ),
    }

    # Filter to only use valid methods
    valid_methods = [(name, method_funcs[name]) for name in methods if name in method_funcs]
    n_methods = len(valid_methods)
    
    if n_methods == 0:
        print("No valid methods specified.")
        return None, [], None
    
    # Calculate grid dimensions
    cols_per_row = min(n_methods, max_cols)
    rows_per_file = 1 #int(np.ceil(n_methods / cols_per_row))
    total_rows = n_files * rows_per_file
    total_cols = cols_per_row
    
    # Create figure
    fig, axs = plt.subplots(total_rows, total_cols, figsize=(5*total_cols, 4*rows_per_file*n_files))
    
    # Ensure axs is a 2D array even for single plot
    if total_rows == 1 and total_cols == 1:
        axs = np.array([[axs]])
    elif total_rows == 1:
        axs = axs.reshape(1, -1)
    elif total_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Get directory name
    dir_short_name = os.path.basename(audio_directory)
    
    # Add a main title
    fig.suptitle(f"PSD Analysis Methods Comparison - Dir: {dir_short_name}", fontsize=16)
    
    # Add instruction text
    instruction_text = fig.text(0.08, 0.94, 
        'LEFT-CLICK: Select peak\n'
        'RIGHT-CLICK: Pair selected peaks\n'
        'DOUBLE-CLICK: Remove point/pair\n'
        'C key: Clear selections\n'
        'R key: Reset all\n'
        'D key: Toggle db scale view', 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', alpha=0.7),
        fontsize=10)

    # Create interactive plots
    plots = []
    
    for file_idx, filename in enumerate(audio_files):
        file_path = os.path.join(audio_directory, filename)
        base_filename = os.path.splitext(filename)[0]
        
        # Starting row for this file
        start_row = file_idx * rows_per_file
        # Process each method for this file
        for method_idx, (method_name, method_func) in enumerate(valid_methods):
            # Calculate row and column for this plot
            row = start_row + (method_idx // cols_per_row)
            col = method_idx % cols_per_row
            
            # Skip if we're out of grid bounds
            if row >= total_rows or col >= total_cols:
                continue
                
            # Calculate PSD using the current method
            try:
                frequencies, psd = method_func(file_path)

                # Check for NaN or Inf values
                if np.any(np.isnan(psd)) or np.any(np.isinf(psd)):
                    print(f"Warning: NaN or Inf values detected in {method_name} for {filename}")
                    psd = np.nan_to_num(psd, nan=1e-10, posinf=1.0, neginf=1e-10)
                
                # Ensure all values are positive for log scale
                psd = np.maximum(psd, 1e-10)
                
                # Create plot with the results
                ax = axs[row, col]
                plot = EnhancedInteractiveHarmonicPlot(
                    frequencies, psd, 
                    f"{base_filename} ({method_name})", 
                    ax, 
                    max_pairs=max_pairs,
                    is_db_scale=use_db_scale,  # Use the parameter here
                    peak_fmin=peak_fmin, 
                    peak_fmax=peak_fmax,
                    plot_fmin=plot_fmin, 
                    plot_fmax=plot_fmax,
                    height_percentile=height_percentile,
                    prominence_factor=prominence_factor,
                    min_width=min_width,
                    method_name=method_name, 
                    top_padding_db=10
                )
                plots.append(plot)

            except Exception as e:
                print(f"Error from compare_methods with {filename}, method {method_name}: {e}")
                axs[row, col].text(0.5, 0.5, f"Error:\n{e}", transform=axs[row, col].transAxes,
                                 horizontalalignment='center', verticalalignment='center')
                axs[row, col].set_title(f"{base_filename} - {method_name} - Failed")

    # Hide any unused subplots
    for row in range(total_rows):
        for col in range(total_cols):
            if row >= n_files * rows_per_file or (row % rows_per_file * cols_per_row + col >= n_methods):
                if row < axs.shape[0] and col < axs.shape[1]:
                    axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, right=0.85, hspace=0.4, wspace=0.3)
    
    # Add a custom save button
    save_ax = fig.add_axes([0.92, 0.01, 0.07, 0.05])
    save_button = Button(save_ax, 'Save')
    
    daily_dir = jelfun.make_daily_directory()
    
    def save_callback(event):
        fig_path, data_path = save_figure_with_timestamp(fig, plots, 
                                                        base_filename=f"psd_methods_comparison_{n_fft}", 
                                                        output_directory=f"{daily_dir}/jellyfish_dynamite")
        print(f"Plot saved successfully!")
    
    save_button.on_clicked(save_callback)
    
    # Create specific save function for this figure
    def save_this_figure(base_filename=f"psd_methods_comparison_{n_fft}", output_directory=None):
        return save_figure_with_timestamp(fig, plots, base_filename, output_directory)
    
    method_names_str = ", ".join(methods)
    
    print(f"Multi-method PSD analysis ready!")
    print(f"Analyzing {n_files} files with methods: {method_names_str}")
    print(f"Peak detection range: {peak_fmin}-{peak_fmax} Hz")
    print(f"Plot display range: {plot_fmin}-{plot_fmax} Hz")
    print("- Left-click on peaks to select them")
    print("- Right-click to form pairs from the last two selected peaks")
    print("- Double-click to remove a point or pair")  
    print("- Press 'c' to clear current selections")
    print("- Press 'r' to reset all pairs and selections")
    
    return fig, plots, save_this_figure, dir_short_name



# ==================== HTML EXPORT FUNCTIONS ====================

def create_interactive_html_plots(fig, plots, base_filename="psd_analysis", output_directory=None):
    """Create an interactive HTML visualization using Plotly with working JavaScript."""

    # Create a custom JSON encoder for NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.number):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Use daily_dir as the default base directory if none provided
    if output_directory is None:
        daily_dir = jelfun.make_daily_directory()
        output_directory = f"{daily_dir}/jellyfish_dynamite_html"
    
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Determine grid dimensions
    total_plots = len(plots)
    n_cols = n_methods
    n_rows = n_files
    
    # Calculate vertical spacing dynamically
    if n_rows > 1:
        v_spacing = min(0.15, 0.2 / (n_rows - 1))
    else:
        v_spacing = 0.15
    
    # Create plotly figure with subplots
    plotly_fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[getattr(plot, 'filename', f"Plot {i+1}") for i, plot in enumerate(plots)],
        vertical_spacing=v_spacing,
        horizontal_spacing=0.08
    )
    
    # Extract plot data for JavaScript
    plot_data = []
    for i, plot in enumerate(plots):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        plot_info = {
            'plot_index': i,
            'row': row,
            'col': col,
            'filename': getattr(plot, 'filename', f"Plot {i+1}"),
            'click_tolerance': getattr(plot, 'click_tolerance', 333),
            'max_pairs': getattr(plot, 'max_pairs', 3),
            'colors': getattr(plot, 'colors', ['red', 'green', 'purple', 'orange', 'brown']),
            'frequencies': plot.frequencies.tolist() if hasattr(plot.frequencies, 'tolist') else list(plot.frequencies),
            
            #'psd_db': plot.psd_db.tolist() if hasattr(plot.psd_db, 'tolist') else list(plot.psd_db),
            #'peak_freqs': plot.peak_freqs.tolist() if hasattr(plot.peak_freqs, 'tolist') else list(plot.peak_freqs),
            #'peak_powers': plot.peak_powers.tolist() if hasattr(plot.peak_powers, 'tolist') else list(plot.peak_powers),
            
            # FIXED: Always provide both scales
            'psd_linear': plot.psd_linear.tolist(),  # Linear scale
            'psd_db': plot.psd_db.tolist(),          # dB scale

            'peak_freqs': plot.peak_freqs.tolist(),
            
            # FIXED: Calculate peak powers for both scales
            'peak_powers_linear': [plot.psd_linear[np.argmin(np.abs(plot.frequencies - f))] for f in plot.peak_freqs],
            'peak_powers_db': [plot.psd_db[np.argmin(np.abs(plot.frequencies - f))] for f in plot.peak_freqs],

            'selected_peaks': [],
            'pairs': [],
            'current_scale': 'db'  # Add scale state tracking




        }
        plot_data.append(plot_info)
        
        # Add main PSD curve
        plotly_fig.add_trace(
            go.Scatter(
                x=plot_info['frequencies'],
                y=plot_info['psd_db'],
                mode='lines',
                name=f"psd_{i}",
                line=dict(color='black', width=1.5),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add detected peaks
        plotly_fig.add_trace(
            go.Scatter(
                x=plot_info['peak_freqs'],
                y=plot_info['peak_powers'],
                mode='markers',
                name=f"peaks_{i}",
                marker=dict(color='gray', size=5, opacity=0.7),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Set axis properties
        if hasattr(plot, 'plot_fmin') and hasattr(plot, 'plot_fmax'):
            plotly_fig.update_xaxes(range=[plot.plot_fmin, plot.plot_fmax], row=row, col=col)
        plotly_fig.update_xaxes(title_text="Frequency (Hz)", row=row, col=col)
        plotly_fig.update_yaxes(title_text="PSD (dB)", row=row, col=col)
    
    # Update main layout
    main_title = "Dir: {dir_name} - Interactive PSD Analysis"
    plotly_fig.update_layout(
        title=main_title,
        height=max(600, 500 * n_rows),
        width=500 * n_cols,
        template="plotly_white",
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Create working JavaScript for interactivity with full visual features
    js_code = f"""

<script>
var plotData = {json.dumps(plot_data, cls=NumpyEncoder)};
var lastClickTime = 0;
var doubleClickTime = 0;

function findNearestPeak(plotIndex, clickX) {{
    var plot = plotData[plotIndex];
    var peaks = plot.peak_freqs;
    var tolerance = plot.click_tolerance;
    var minDist = Infinity;
    var nearestIdx = -1;
    
    for (var i = 0; i < peaks.length; i++) {{
        var dist = Math.abs(peaks[i] - clickX);
        if (dist < minDist && dist < tolerance) {{
            minDist = dist;
            nearestIdx = i;
        }}
    }}
    return nearestIdx >= 0 ? nearestIdx : null;
}}

function selectPeak(plotIndex, peakIdx) {{
    var plot = plotData[plotIndex];
    var freq = plot.peak_freqs[peakIdx];
    
    if (plot.selected_peaks.indexOf(freq) === -1) {{
        plot.selected_peaks.push(freq);
        console.log('Selected peak at ' + freq.toFixed(1) + ' Hz');
        updatePlot(plotIndex);
    }}
}}

function removePeak(plotIndex, peakIdx) {{
    var plot = plotData[plotIndex];
    var freq = plot.peak_freqs[peakIdx];
    
    // Remove from selected peaks
    var idx = plot.selected_peaks.indexOf(freq);
    if (idx !== -1) {{
        plot.selected_peaks.splice(idx, 1);
    }}
    
    // Remove pairs containing this frequency
    plot.pairs = plot.pairs.filter(function(pair) {{
        return Math.abs(pair.f0 - freq) >= plot.click_tolerance && 
            Math.abs(pair.f1 - freq) >= plot.click_tolerance;
    }});
    
    console.log('Removed peak at ' + freq.toFixed(1) + ' Hz');
    updatePlot(plotIndex);
}}

function createPair(plotIndex) {{
    var plot = plotData[plotIndex];
    
    if (plot.selected_peaks.length >= 2 && plot.pairs.length < plot.max_pairs) {{
        var f1 = plot.selected_peaks[plot.selected_peaks.length - 2];
        var f2 = plot.selected_peaks[plot.selected_peaks.length - 1];
        var f0 = Math.min(f1, f2);
        var f1_sorted = Math.max(f1, f2);
        var color = plot.colors[plot.pairs.length % plot.colors.length];
        
        plot.pairs.push({{
            f0: f0,
            f1: f1_sorted,
            color: color
        }});
        
        // Remove from selected
        plot.selected_peaks = plot.selected_peaks.filter(function(peak) {{
            return Math.abs(peak - f0) >= plot.click_tolerance && 
                Math.abs(peak - f1_sorted) >= plot.click_tolerance;
        }});
        
        console.log('Created pair: ' + f0.toFixed(1) + ' Hz and ' + f1_sorted.toFixed(1) + ' Hz, ratio: ' + (f1_sorted/f0).toFixed(3));
        updatePlot(plotIndex);
    }}
}}

function clearSelections(plotIndex) {{
    plotData[plotIndex].selected_peaks = [];
    updatePlot(plotIndex);
}}

function resetAll(plotIndex) {{
    plotData[plotIndex].selected_peaks = [];
    plotData[plotIndex].pairs = [];
    console.log('Reset all for plot ' + plotIndex);
    updatePlot(plotIndex);
}}

function connectAll(plotIndex) {{
    var plot = plotData[plotIndex];
    
    // Get all nodes that should be in the fully connected graph:
    // 1. Currently selected peaks 
    // 2. Peaks that are already in existing pairs
    var allNodes = new Set();
    
    // Add currently selected peaks
    for (var i = 0; i < plot.selected_peaks.length; i++) {{
        allNodes.add(plot.selected_peaks[i]);
    }}
    
    // Add peaks from existing pairs
    for (var i = 0; i < plot.pairs.length; i++) {{
        allNodes.add(plot.pairs[i].f0);
        allNodes.add(plot.pairs[i].f1);
    }}
    
    var nodes = Array.from(allNodes).sort(function(a, b) {{ return a - b; }});
    
    if (nodes.length < 2) {{
        console.log('Need at least 2 points to connect');
        return;
    }}
    
    console.log('Creating fully connected graph with ' + nodes.length + ' nodes: ' + nodes.map(n => n.toFixed(1)).join(', '));
    
    // Create pairs for all combinations that don't already exist
    var existingPairSet = new Set();
    for (var i = 0; i < plot.pairs.length; i++) {{
        var pair = plot.pairs[i];
        var key1 = pair.f0 + '-' + pair.f1;
        var key2 = pair.f1 + '-' + pair.f0;
        existingPairSet.add(key1);
        existingPairSet.add(key2);
    }}
    
    var newPairsAdded = 0;
    for (var i = 0; i < nodes.length; i++) {{
        for (var j = i + 1; j < nodes.length; j++) {{
            if (plot.pairs.length >= plot.max_pairs) {{
                console.log('Reached maximum pairs limit');
                break;
            }}
            
            var f0 = Math.min(nodes[i], nodes[j]);
            var f1 = Math.max(nodes[i], nodes[j]);
            var pairKey = f0 + '-' + f1;
            
            // Only add if this pair doesn't already exist
            if (!existingPairSet.has(pairKey)) {{
                var color = plot.colors[plot.pairs.length % plot.colors.length];
                
                plot.pairs.push({{
                    f0: f0,
                    f1: f1,
                    color: color
                }});
                
                existingPairSet.add(pairKey);
                existingPairSet.add(f1 + '-' + f0);
                newPairsAdded++;
            }}
        }}
        if (plot.pairs.length >= plot.max_pairs) break;
    }}
    
    // DON'T clear selected peaks - keep them for potential future operations
    // plot.selected_peaks = [];  // REMOVED THIS LINE
    
    console.log('Added ' + newPairsAdded + ' new connections to create fully connected graph');
    console.log('Total pairs: ' + plot.pairs.length);
    updatePlot(plotIndex);
}}


// Add scale toggle function
function toggleScale(plotIndex) {{
    var plot = plotData[plotIndex];
    var fig = document.getElementsByClassName('js-plotly-plot')[0];
    
    // Toggle scale state
    plot.current_scale = plot.current_scale === 'db' ? 'linear' : 'db';
    
    // Update main PSD trace
    var psdTraceIndex = plotIndex * 2;
    var newYData = plot.current_scale === 'db' ? plot.psd_db : plot.psd_linear;
    
    Plotly.restyle(fig, {{'y': [newYData]}}, [psdTraceIndex]);
    
    // Update peak powers
    plot.peak_powers = plot.current_scale === 'db' ? plot.peak_powers_db : plot.peak_powers_linear;
    
    // FIXED: Update axis label properly
    var yAxisKey = 'yaxis' + (plot.row === 1 && plot.col === 1 ? '' : ((plot.row-1)*3 + plot.col));
    var newYLabel = plot.current_scale === 'db' ? 'PSD (dB)' : 'PSD (linear)';
    
    var layoutUpdate = {{}};
    layoutUpdate[yAxisKey] = {{title: newYLabel}};  // FIXED: Proper nested object
    Plotly.relayout(fig, layoutUpdate);
    
    console.log('Toggled to ' + plot.current_scale + ' scale');
}}


function updatePlot(plotIndex) {{
    var plot = plotData[plotIndex];
    var row = plot.row;
    var col = plot.col;
    var fig = document.getElementsByClassName('js-plotly-plot')[0];
    
    // PRESERVE AXIS RANGES - Get current ranges before any modifications
    var currentLayout = fig.layout;
    var xAxisKey = 'xaxis' + (row === 1 && col === 1 ? '' : ((row-1)*3 + col));
    var yAxisKey = 'yaxis' + (row === 1 && col === 1 ? '' : ((row-1)*3 + col));
    var currentXRange = currentLayout[xAxisKey] ? currentLayout[xAxisKey].range : null;
    var currentYRange = currentLayout[yAxisKey] ? currentLayout[yAxisKey].range : null;
    
    // Remove old interactive traces
    var tracesToRemove = [];
    
    for (var i = 0; i < fig.data.length; i++) {{
        if (fig.data[i].name && (fig.data[i].name.includes('selected_' + plotIndex) || 
                                fig.data[i].name.includes('pair_' + plotIndex) ||
                                fig.data[i].name.includes('vline_' + plotIndex))) {{
            tracesToRemove.push(i);
        }}
    }}
    
    if (tracesToRemove.length > 0) {{
        Plotly.deleteTraces(fig, tracesToRemove);
    }}
    
    // Clear old ratio annotations for this plot only
    var layout = fig.layout;
    if (layout.annotations) {{
        layout.annotations = layout.annotations.filter(function(ann) {{
            return !(ann.plotIndex && ann.plotIndex === plotIndex && ann.isRatio);
        }});
    }} else {{
        layout.annotations = [];
    }}
    
    // Prepare all traces to add in one batch
    var tracesToAdd = [];
    var axisRef = row === 1 && col === 1 ? '' : (row-1)*3 + col;
    var xAxisRef = 'x' + axisRef;
    var yAxisRef = 'y' + axisRef;
    
    // Add selected peaks with blue markers and frequency labels
    if (plot.selected_peaks.length > 0) {{
        var selectedX = [];
        var selectedY = [];
        var selectedText = [];
        
        for (var i = 0; i < plot.selected_peaks.length; i++) {{
            var freq = plot.selected_peaks[i];
            var peakIdx = plot.peak_freqs.findIndex(function(f) {{ 
                return Math.abs(f - freq) < plot.click_tolerance; 
            }});
            
            if (peakIdx >= 0) {{
                selectedX.push(freq);
                selectedY.push(plot.peak_powers[peakIdx]);
                selectedText.push(freq.toFixed(0) + ' Hz');
            }}
        }}
        
        if (selectedX.length > 0) {{
            tracesToAdd.push({{
                x: selectedX,
                y: selectedY,
                mode: 'markers+text',
                marker: {{ color: 'blue', size: 10, line: {{ color: 'black', width: 1 }} }},
                text: selectedText,
                textposition: 'top center',
                textfont: {{ color: 'blue', size: 10 }},
                name: 'selected_' + plotIndex,
                showlegend: false,
                xaxis: xAxisRef,
                yaxis: yAxisRef
            }});
        }}
    }}
    
    // Prepare pair traces in batch
    var pairVerticalLines = {{ x: [], y: [], mode: 'lines', line: {{ color: 'gray', width: 1, dash: 'dash' }}, opacity: 0.7, name: 'vlines_' + plotIndex, showlegend: false, xaxis: xAxisRef, yaxis: yAxisRef }};
    var yMin = Math.min(...plot.psd_db);
    var yMax = Math.max(...plot.psd_db);
    
    // Add pairs with connecting lines and colored vertical lines
    for (var i = 0; i < plot.pairs.length; i++) {{
        var pair = plot.pairs[i];
        var f0Idx = plot.peak_freqs.findIndex(function(f) {{ 
            return Math.abs(f - pair.f0) < plot.click_tolerance; 
        }});
        var f1Idx = plot.peak_freqs.findIndex(function(f) {{ 
            return Math.abs(f - pair.f1) < plot.click_tolerance; 
        }});
        
        if (f0Idx >= 0 && f1Idx >= 0) {{
            var f0Power = plot.peak_powers[f0Idx];
            var f1Power = plot.peak_powers[f1Idx];
            
            // Add to batch vertical lines
            pairVerticalLines.x.push(pair.f0, pair.f0, null, pair.f1, pair.f1, null);
            pairVerticalLines.y.push(yMin, yMax, null, yMin, yMax, null);
            
            // Add connecting line with markers
            tracesToAdd.push({{
                x: [pair.f0, pair.f1],
                y: [f0Power, f1Power],
                mode: 'lines+markers+text',
                line: {{ color: pair.color, width: 2 }},
                marker: {{ color: 'blue', size: 10, line: {{ color: 'black', width: 1 }} }},
                text: [pair.f0.toFixed(0) + ' Hz', pair.f1.toFixed(0) + ' Hz'],
                textposition: 'top center',
                textfont: {{ color: 'blue', size: 10 }},
                name: 'pair_' + plotIndex + '_' + i,
                showlegend: false,
                xaxis: xAxisRef,
                yaxis: yAxisRef
            }});
        }}
    }}
    
    // Add the batched vertical lines if any pairs exist
    if (pairVerticalLines.x.length > 0) {{
        tracesToAdd.push(pairVerticalLines);
    }}
    
    // CREATE TOP-RIGHT RATIO LIST for this plot
    if (plot.pairs.length > 0) {{
        // Create a single compact legend-style annotation
        var ratioTexts = [];
        for (var i = 0; i < plot.pairs.length; i++) {{
            var pair = plot.pairs[i];
            var ratio = pair.f1 / pair.f0;
            ratioTexts.push(pair.f0.toFixed(0) + '→' + pair.f1.toFixed(0) + ' (r:' + ratio.toFixed(2) + ')');
        }}
        
        // Join all ratios into a single multi-line text block
        var combinedText = ratioTexts.join('<br>');
        
        // Calculate position for top-right of this specific subplot
        var xDomain = currentLayout[xAxisKey] ? currentLayout[xAxisKey].domain : [0, 1];
        var yDomain = currentLayout[yAxisKey] ? currentLayout[yAxisKey].domain : [0, 1];
        
        // Position at top-right corner of the subplot
        var listX = xDomain[1] - 0.005; // Very close to right edge
        var listY = yDomain[1] - 0.005; // Very close to top edge
        
        // Create single annotation with all ratios
        layout.annotations.push({{
            x: listX,
            y: listY,
            text: combinedText,
            showarrow: false,
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            bordercolor: 'gray',
            borderwidth: 1,
            font: {{ size: 8, color: 'black' }},
            plotIndex: plotIndex,
            isRatio: true,
            xref: 'paper',
            yref: 'paper',
            xanchor: 'right',
            yanchor: 'top',
            align: 'right'
        }});
    }}
    
    // SINGLE BATCH UPDATE - Add all traces at once, then update layout
    var updatePromise = Promise.resolve();
    
    if (tracesToAdd.length > 0) {{
        updatePromise = Plotly.addTraces(fig, tracesToAdd);
    }}
    
    // Update layout and restore axis ranges in one operation
    var layoutUpdate = {{ annotations: layout.annotations }};
    
    // Restore axis ranges if they were captured
    if (currentXRange) {{
        layoutUpdate[xAxisKey + '.range'] = currentXRange;
    }}
    if (currentYRange) {{
        layoutUpdate[yAxisKey + '.range'] = currentYRange;
    }}
    
    updatePromise.then(function() {{
        return Plotly.relayout(fig, layoutUpdate);
    }}).then(function() {{
        // Console debug info
        if (plot.pairs.length > 0) {{
            console.log('=== Pairs for Plot ' + plotIndex + ' ===');
            for (var i = 0; i < plot.pairs.length; i++) {{
                var pair = plot.pairs[i];
                var ratio = pair.f1 / pair.f0;
                console.log('Pair ' + (i+1) + ': ' + pair.f0.toFixed(1) + ' Hz <-> ' + pair.f1.toFixed(1) + ' Hz (ratio: ' + ratio.toFixed(3) + ')');
            }}
        }}
    }});
}}



document.addEventListener('DOMContentLoaded', function() {{
    var plotDiv = document.getElementsByClassName('js-plotly-plot')[0];
    
    plotDiv.on('plotly_click', function(data) {{
        if (!data.points || data.points.length === 0) return;
        
        var point = data.points[0];
        var clickX = point.x;
        var plotIndex = 0;
        
        // Determine which subplot was clicked
        if (point.data.name && point.data.name.includes('_')) {{
            var parts = point.data.name.split('_');
            if (parts.length > 1) {{
                plotIndex = parseInt(parts[1]) || 0;
            }}
        }}
        
        var currentTime = Date.now();
        var timeDiff = currentTime - lastClickTime;
        var isDoubleClick = timeDiff < 300;
        
        // Check for Ctrl key modifier
        var isCtrlClick = data.event && (data.event.ctrlKey || data.event.metaKey);
        
        if (isDoubleClick) {{
            doubleClickTime = currentTime;
        }}
        
        lastClickTime = currentTime;
        
        var peakIdx = findNearestPeak(plotIndex, clickX);
        
        if (peakIdx !== null) {{
            if (isCtrlClick) {{
                // Ctrl+click to remove peak
                removePeak(plotIndex, peakIdx);
            }} else if (isDoubleClick) {{
                // Double-click to remove peak (alternative method)
                removePeak(plotIndex, peakIdx);
            }} else {{
                // Regular click to select peak
                selectPeak(plotIndex, peakIdx);
            }}
        }}
    }});
    
    plotDiv.addEventListener('contextmenu', function(e) {{
        e.preventDefault();
        
        // Simple right-click handler for pair creation
        var rect = plotDiv.getBoundingClientRect();
        var plotIndex = 0; // For simplicity, apply to first plot
        createPair(plotIndex);
        
        return false;
    }});
    
    document.addEventListener('keypress', function(event) {{
        var key = event.key.toLowerCase();
        
        for (var i = 0; i < plotData.length; i++) {{
            switch(key) {{
                case 'c':
                    clearSelections(i);
                    break;
                case 'r':
                    resetAll(i);
                    break;
                case 'a':
                    connectAll(i);
                    break;
            }}
        }}
    }});
    
    console.log('Interactive PSD analysis ready');
    console.log('Left-click: select peak, Ctrl+click: remove peak, Double-click: remove peak');
    console.log('Right-click: create pair, Keys: c=clear, r=reset, a=fully connect node graph, d=toggle db scale view');
}});
</script>

"""
    
    # Define the HTML file path
    html_filename = f"{base_filename}_{jelfun.get_timestamp()}.html"
    html_path = os.path.join(output_directory, html_filename)
    
    # Generate the HTML content
    html_str = plotly_fig.to_html(
        include_plotlyjs='cdn',
        full_html=True,
        config={
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': True,
            'displaylogo': False,
            'editable': False,
            'staticPlot': False,
        }
    )
    
    # Add normal cursor styling
    cursor_style = '<style>body { cursor: default !important; } .js-plotly-plot .plotly { cursor: default !important; } .js-plotly-plot .plotly .cursor-crosshair { cursor: default !important; } .plotly { cursor: default !important; } * { cursor: default !important; }</style>'
    html_str = html_str.replace('<head>', f'<head>{cursor_style}')
    
    # Insert JavaScript
    html_str = html_str.replace('</body>', f'{js_code}</body>')
    
    # Save to file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    
    print(f"Interactive HTML saved to {html_path}")
    
    # Save data files
    data_filename = f"{base_filename}_{jelfun.get_timestamp()}_pairdata.json"
    data_path = os.path.join(output_directory, data_filename)
    
    export_data = []
    for plot in plots:
        file_data = {
            'filename': getattr(plot, 'filename', 'Untitled'),
            'method': getattr(plot, 'method_name', 'Unknown'),
            'pairs': []
        }
        if hasattr(plot, 'pairs'):
            for pair in plot.pairs:
                if 'f0' in pair and 'f1' in pair:
                    file_data['pairs'].append({
                        'f0': float(pair['f0']),
                        'f1': float(pair['f1']),
                        'ratio': float(pair['f1'] / pair['f0'])
                    })
        export_data.append(file_data)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, cls=NumpyEncoder)
    
    graph_filename = f"{base_filename}_{jelfun.get_timestamp()}_graphdata.json"
    graph_path = os.path.join(output_directory, graph_filename)
    
    graph_export_data = []
    for plot in plots:
        graph_data = {'nodes': [], 'edges': []}
        if hasattr(plot, 'get_graph_data'):
            try:
                raw_graph = plot.get_graph_data()
                if 'nodes' in raw_graph:
                    for node in raw_graph['nodes']:
                        safe_node = {}
                        for key, value in node.items():
                            if hasattr(value, 'item'):
                                safe_node[key] = value.item()
                            elif isinstance(value, (np.ndarray, np.number)):
                                safe_node[key] = float(value)
                            else:
                                safe_node[key] = value
                        graph_data['nodes'].append(safe_node)
                
                if 'edges' in raw_graph:
                    for edge in raw_graph['edges']:
                        safe_edge = {}
                        for key, value in edge.items():
                            if hasattr(value, 'item'):
                                safe_edge[key] = value.item()
                            elif isinstance(value, (np.ndarray, np.number)):
                                safe_edge[key] = float(value)
                            else:
                                safe_edge[key] = value
                        graph_data['edges'].append(safe_edge)
            except Exception as e:
                print(f"Error processing graph data: {e}")
        
        file_data = {
            'filename': getattr(plot, 'filename', 'Untitled'),
            'method': getattr(plot, 'method_name', 'Unknown'),
            'graph': graph_data
        }
        graph_export_data.append(file_data)
    
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_export_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"Data saved to: {data_path}")
    print(f"Graph data saved to: {graph_path}")
    
    try:
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
        print(f"Opening HTML file in browser...")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open manually: {html_path}")
    
    return html_path, data_path, graph_path




# ==================== SAVE FUNCTIONS ====================

def save_figure_with_timestamp(fig, plots, base_filename="psd_anal", output_directory=None):
    """Save a figure with timestamp as part of the filename for versioning."""
    if output_directory is None:
        daily_dir = jelfun.make_daily_directory()
        output_directory = f"{daily_dir}/jellyfish_dynamite_plots"
    
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Save figure with timestamp
    fig_filename = f"{base_filename}_{jelfun.get_timestamp()}_fig.png"
    fig_path = os.path.join(output_directory, fig_filename)
    
    # Save the figure
    fig.savefig(fig_path, bbox_inches='tight', dpi=200)
    
    # Save pair data as JSON
    data_filename = f"{base_filename}_{jelfun.get_timestamp()}_pairdata.json"
    data_path = os.path.join(output_directory, data_filename)
    
    # Export data
    export_data = []
    for plot in plots:
        file_data = {
            'filename': plot.filename,
            'pairs': []
        }
        for pair in plot.pairs:
            file_data['pairs'].append({
                'f0': float(pair['f0']),
                'f1': float(pair['f1']),
                'f1_f0_ratio': float(pair['f1'] / pair['f0'])
            })
        export_data.append(file_data)
    
    # Save the data
    with open(data_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Figure saved to {fig_path}")
    print(f"Data saved to {data_path}")

    return fig_path, data_path


def save_jellyfish_jinja(template_vars, template_name, base_filename="psd_analysis", output_directory=None):
    """Agnostic Jinja templating function - works with any template and data."""
    
    # Custom JSON encoder for NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.number):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Use daily_dir as default if none provided
    if output_directory is None:
        try:
            daily_dir = jelfun.make_daily_directory()
            output_directory = f"{daily_dir}/jellyfish_dynamite_html"
        except:
            output_directory = "html_output"
    
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Load template file
    if not os.path.exists(template_name):
        print(f"Template not found: {template_name}")
        print(f"Current directory: {os.getcwd()}")
        return None, None, None
    
    with open(template_name, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Create Jinja2 template and render
    template = Template(template_content)
    html_str = template.render(**template_vars)
    
    # Define file paths
    html_filename = f"{base_filename}_{jelfun.get_timestamp()}.html"
    html_path = os.path.join(output_directory, html_filename)
    
    # Save HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    
    print(f"HTML saved to: {html_path}")
    
    try:
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
        print(f"Opening HTML file in browser...")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open manually: {html_path}")
    
    return html_path





# PLOTLY

def prepare_plotly_template_vars(plots, methods=None, dir_name=None, use_db_scale=True):
    """Prepare template variables specifically for Plotly templates with dual scale support."""

    # Custom JSON encoder for NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.number):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Calculate grid dimensions from plots
    total_plots = len(plots)
    filenames = set()
    methods_set = set()
    for plot in plots:
        base_name = plot.filename.split(' (')[0] if ' (' in plot.filename else plot.filename
        filenames.add(base_name)
        if hasattr(plot, 'method_name'):
            methods_set.add(plot.method_name)

    n_files = len(filenames)
    n_methods = len(methods_set) if methods_set else total_plots
    n_rows = n_files
    n_cols = n_methods
    
    # Calculate vertical spacing
    if n_rows > 1:
        v_spacing = min(0.15, 0.2 / (n_rows - 1))
    else:
        v_spacing = 0.15
    
    # Create plotly figure with subplots
    plotly_fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[getattr(plot, 'filename', f"Plot {i+1}") for i, plot in enumerate(plots)],
        vertical_spacing=v_spacing,
        horizontal_spacing=0.08
    )
    
    # Helper function to safely convert arrays
    def safe_tolist(arr):
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        elif isinstance(arr, (list, tuple)):
            return list(arr)
        else:
            return arr
    
    # Process each plot
    for i, plot in enumerate(plots):
        file_idx = i // n_methods
        method_idx = i % n_methods
        row = file_idx + 1
        col = method_idx + 1
        
        
        frequencies = safe_tolist(plot.frequencies)        
        
        # Get both linear and dB versions of the data
        linear_psd = safe_tolist(plot.original_psd if hasattr(plot, 'original_psd') else plot.psd)
        db_psd = safe_tolist(plot.psd_db)
        
        # Also get peak data in both scales
        peak_freqs = safe_tolist(plot.peak_freqs)

        # Calculate peak powers for both scales
        peak_powers_linear = []
        peak_powers_db = []
        for peak_freq in plot.peak_freqs:
            freq_idx = np.argmin(np.abs(plot.frequencies - peak_freq))
            peak_powers_linear.append(float(plot.psd_linear[freq_idx]))
            peak_powers_db.append(float(plot.psd_db[freq_idx]))


        # Use the scale parameter to determine starting data
        starting_psd = db_psd if use_db_scale else linear_psd
        starting_peak_powers = peak_powers_db if use_db_scale else peak_powers_linear

        # Add main PSD curve with both scales stored
        plotly_fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=starting_psd,
                mode='lines',
                name=f"psd_{i}",
                line=dict(color='black', width=1.5),
                showlegend=False,
                # Store both scales in customdata/meta
                #customdata=[{
                meta={
                    'linear_psd': linear_psd,
                    'db_psd': db_psd,
                    'scale_type': 'main_trace'
                }#] * len(frequencies)
            ),
            row=row, col=col
        )
        
        # Add detected peaks with both scales stored
        plotly_fig.add_trace(
            go.Scatter(
                x=peak_freqs,
                y=starting_peak_powers,
                mode='markers',
                name=f"peaks_{i}",
                marker=dict(color='gray', size=5, opacity=0.7),
                showlegend=False,
                # Store both scales in customdata/meta
                #customdata=[{
                meta={
                    'linear_powers': peak_powers_linear,
                    'db_powers': peak_powers_db,
                    'scale_type': 'peak_trace'
                }#] * len(peak_freqs)
            ),
            row=row, col=col
        )
        
        # Set axis properties
        if hasattr(plot, 'plot_fmin') and hasattr(plot, 'plot_fmax'):
            plotly_fig.update_xaxes(range=[plot.plot_fmin, plot.plot_fmax], row=row, col=col)
        plotly_fig.update_xaxes(title_text="Frequency (Hz)", row=row, col=col)
        plotly_fig.update_yaxes(title_text="PSD (dB)", row=row, col=col)  # Start with dB
    
    # Update main layout
    plotly_fig.update_layout(
        title=f"Interactive PSD Analysis - Dir: {dir_name}",
        height=max(600, 500 * n_rows),
        width=max(800, 300 * n_cols),
        template="plotly_white",
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Convert to JSON
    fig_json = plotly_fig.to_json()
    fig_dict = json.loads(fig_json)
    

        
    subplot_titles = [getattr(plot, 'filename', f"Plot {i+1}") for i, plot in enumerate(plots)]

    # Return template variables for Plotly
    return {
        'PLOT_ID': f"plot_{jelfun.get_timestamp()}",
        'PLOT_HEIGHT': max(600, 500 * n_rows),
        'PLOT_WIDTH': max(800, 300 * n_cols),
        'PLOT_DATA': json.dumps(fig_dict['data']),
        'LAYOUT_DATA': json.dumps(fig_dict['layout']),
        'SUBPLOT_TITLES': json.dumps(subplot_titles), 
        'DIR_NAME': dir_name, 
        'USE_DB_SCALE': 'true' if use_db_scale else 'false',

    }



def save_jellyfish_plotly(plots, base_filename="psd_analysis_plotly", output_directory=None, methods=None, dir_name=None, use_db_scale=True, **kwargs):
    """Convenience wrapper for saving Plotly plots using Jinja templates."""
    n_fft = kwargs.get('n_fft')
    nfft_suffix = f"_nfft{n_fft}" if n_fft else ""

    # Prepare Plotly-specific template variables
    template_vars = prepare_plotly_template_vars(plots, methods, dir_name, use_db_scale)

    # Use the Plotly template
    template_name = "jellyfish_dynamite_plotly.html"
    
    # Call the agnostic Jinja function
    html_path = save_jellyfish_jinja(template_vars, template_name, base_filename, output_directory)
    
    # Save additional data files (keeping existing functionality)
    if output_directory is None:
        daily_dir = jelfun.make_daily_directory()
        output_directory = f"{daily_dir}/jellyfish_dynamite_html"
    
    # Save pair and graph data (existing code from original function)
    data_filename = f"{dir_name}_{base_filename}{nfft_suffix}_{jelfun.get_timestamp()}_pairdata.json"
    data_path = os.path.join(output_directory, data_filename)
    
    export_data = []
    for plot in plots:
        file_data = {
            'filename': getattr(plot, 'filename', 'Untitled'),
            'method': getattr(plot, 'method_name', 'Unknown'),
            'pairs': []
        }
        if hasattr(plot, 'pairs'):
            for pair in plot.pairs:
                if 'f0' in pair and 'f1' in pair:
                    file_data['pairs'].append({
                        'f0': float(pair['f0']),
                        'f1': float(pair['f1']),
                        'ratio': float(pair['f1'] / pair['f0'])
                    })
        export_data.append(file_data)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    # Save graph data
    graph_filename = f"{dir_name}_{base_filename}{nfft_suffix}_{jelfun.get_timestamp()}_graphdata.json"
    graph_path = os.path.join(output_directory, graph_filename)
    
    graph_export_data = []
    for plot in plots:
        graph_data = {'nodes': [], 'edges': []}
        if hasattr(plot, 'get_graph_data'):
            try:
                raw_graph = plot.get_graph_data()
                if 'nodes' in raw_graph:
                    for node in raw_graph['nodes']:
                        safe_node = {}
                        for key, value in node.items():
                            if hasattr(value, 'item'):
                                safe_node[key] = value.item()
                            elif isinstance(value, (np.ndarray, np.number)):
                                safe_node[key] = float(value)
                            else:
                                safe_node[key] = value
                        graph_data['nodes'].append(safe_node)
                
                if 'edges' in raw_graph:
                    for edge in raw_graph['edges']:
                        safe_edge = {}
                        for key, value in edge.items():
                            if hasattr(value, 'item'):
                                safe_edge[key] = value.item()
                            elif isinstance(value, (np.ndarray, np.number)):
                                safe_edge[key] = float(value)
                            else:
                                safe_edge[key] = value
                        graph_data['edges'].append(safe_edge)
            except Exception as e:
                print(f"Error processing graph data: {e}")
        
        file_data = {
            'filename': getattr(plot, 'filename', 'Untitled'),
            'method': getattr(plot, 'method_name', 'Unknown'),
            'graph': graph_data
        }
        graph_export_data.append(file_data)
    
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_export_data, f, indent=2)
    
    print(f"Data saved to: {data_path}")
    print(f"Graph data saved to: {graph_path}")
    
    return html_path, data_path, graph_path



def main():
    try:
        import ipympl
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'widget')
    except (ImportError, AttributeError):
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'notebook')
        except:
            matplotlib.use('TkAgg')
            plt.ion()

    main_slicedir = all_slicedirs[68]  # Update this path

    nfft = 1024
    methods = ["FFT", "CQT", "Multi-Res", "Chirplet Zero"]
    
    selected_files = select_audio_files(
        main_slicedir,
        range_start=0,
        range_end=None
    )
    
    # MATPLOTLIB PYTHON PLOTS
    fig, plots, _, dir_short_name = compare_methods_psd_analysis(
        audio_directory=main_slicedir,
        max_cols=len(methods), 
        max_pairs=10,
        n_fft=nfft,
        peak_fmin=100,
        peak_fmax=5000,
        plot_fmin=100,
        plot_fmax=5000,
        selected_files=selected_files,
        methods=methods,
        use_db_scale=False
    )

    # HTML PLOTLY PLOTS
    if fig is not None:
        plt.show()
        html_path, data_path, graph_path = save_jellyfish_plotly(
            plots, 
            base_filename=f"psd_analysis_plotly_{dir_short_name}",  # USE dir_short_name
            methods=methods,
            dir_name=dir_short_name,  # PASS dir_short_name
            use_db_scale=False 

        )
    else:
        print("No audio files found or analysis failed.")

if __name__ == "__main__":
    main()