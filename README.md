<small>

# Jellyfish Dynamite - A Clear-Box Interactive PSD Analysis Tool

A Flask web app for analyzing and comparing audio files using multiple spectral analysis methods and interactive visualization.

## Features

- **Multiple Analysis Methods**: FFT, CQT, Multi-Resolution, Chirplet Transform, and Wavelet-based approaches
- **Interactive Web Interface**: Upload audio files and configure analysis parameters
- **Real-time Peak Detection**: Automatic detection of spectral peaks with adjustable sensitivity
- **Interactive Plotting**: Click to select peaks, create harmonic pairs, and analyze frequency relationships
- **Dual Scale Support**: Toggle between linear and dB scales
- **HTML Export**: Generate standalone interactive HTML visualizations using Plotly
- **Data Export**: Save analysis results or plot images as CSV, JSON or JPG files

## Installation

### Prerequisites

- **Python 3.9 or higher, probably** (developed and tested on Python 3.12.10)
- pip package manager
- Git (for cloning the repository)

### Quick Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/laelume/jellyfish_dynamite.git
   ```

   ```bash
   cd jellyfish_dynamite
   ```
   
   **Or download ZIP:**
   - Click "Code" (top right) → "Download ZIP" on GitHub
   - Extract and navigate to the folder

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   ```

   # On Windows:
   ```bash
   venv\Scripts\activate
   ```

   # On Mac/Linux:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python jelly_app.py
   ```

5. **Open browser** and go to:
   ```
   http://localhost:5000
   ```

6. **Analyze Audio**

The `test_audio/` folder contains sample audio files to test the application. 

- Click "Choose Files"
- Navigate to the test_audio/ folder
- Select one or more test files
- Click "Analyze"


### Installation Options

**Option 1: Flask Web App Standalone**
```bash
pip install -r requirements.txt
```

**Option 2: Flask + Jupyter Notebook Support**
```bash
pip install -r requirements-jupyter.txt
```

## Three Ways to Use Jellyfish Dynamite

### 1. Flask Web App (Featured)
**Use Case**: Quick install, batch processing, easy file upload
```bash
python app.py
```
# Go to http://localhost:5000

- Web-based file upload interface
- Parameter configuration through forms
- Automatic HTML generation and browser opening

### 2. Jupyter Notebook Analysis 
**Use Case**: Low-level parameter adjustment, interactive development, data exploration
```bash
jupyter notebook
```
```bash
# Import functions or use provided notebooks
```
- Interactive matplotlib plots with full controls
- Step-by-step analysis workflow
- Iterative parameter adjustment

### 3. Standalone Python Script 
**Use Case**: Command-line use, automated processing, batch analysis
```bash
python jellyfish_plotly_browser.py
```
```bash
# Analyzes files in specified directory
```
- Direct file processing from directories
- Generates HTML output automatically
- No web server required

## Quick Start


### A. Windows

1. **Download Python 3.9+** from [python.org](https://python.org) 
   - ⚠️ **Important**: Check "Add Python to PATH" during installation
2. **Download this project** as ZIP and extract it
3. **Double-click** `jelly_roll.bat`
4. **Wait** for installation (2-3 minutes)
5. **Open browser** to http://localhost:5000 when ready


### B. Mac/Linux

1. **Install Python 3.9+**:
   - **Mac Option A**: Download from [python.org](https://python.org)
   - **Mac Option B**: Use Homebrew: `brew install python3`
   - **Linux**: Use package manager: `sudo apt install python3` (Ubuntu) or `sudo yum install python3` (CentOS)
2. **Download this project** as ZIP and extract it
3. **Open Terminal** in the project folder
4. **IMPORTANT!!! Make script executable**: `chmod +x jelly_roll.sh`
5. **Run the installer**: `./jelly_roll.sh`
6. **Wait** for installation (2-3 minutes)
7. **Open browser** to http://localhost:5000 when ready

**If the above doesn't work:**
- **Windows**: Open Command Prompt in project folder, run the commands from "For Developers" section
- **Mac/Linux**: Open Terminal in project folder, run the commands from "For Developers" section


### C. For Developers

1. **Clone or download the project files:**
   
   **Option A: Using Git (Recommended)**
   ```bash
   git clone https://github.com/laelume/jellyfish_dynamite.git
   cd jellyfish_dynamite
   ```
   
   **Option B: Download ZIP**
   - Go to GitHub repo
   - Click green "Code" button
   - Select "Download ZIP"
   - Extract ZIP file to desired location
   - Open terminal/command prompt in extracted folder

2. **Check Python version**:
   ```bash
   python --version  # Should be 3.9 or higher
   ```

3. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```
   
   ```bash
   venv\Scripts\activate  # Windows
   ```
   ```bash
   source venv/bin/activate  # Mac/Linux
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Flask application**:
   ```bash
   python jelly_app.py
   ```

6. **Open browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage Guide

### Web Interface

1. **Upload Audio Files**
   - Select one or more audio files (.wav, .mp3, .flac)
   - Files are processed in the order uploaded
   - More files + more methods = more laggy

2. **Configure Analysis Parameters**
   - **Analysis Methods**: Choose from FFT, CQT, Multi-Res, Chirplet, Wavelet variants
   - **FFT Size (n_fft)**: Window size for spectral analysis (512, 1024, 2048, 4096)
   - **Peak Detection Range**: Set frequency bounds for peak detection (Hz)
   - **Plot Display Range**: Set frequency bounds for visualization (Hz)
   - **Scale**: Choose between dB and linear amplitude scales
   - **Directory Name**: Custom name for output files and save options

3. **Submit Analysis**
   - Click "Analyze" to process files
   - Results open automatically in browser

### Interactive Controls

Once analysis is complete, use these controls in the interactive plot:

#### Mouse Controls
- **Left Click**: Select a spectral peak (blue dot with frequency label)
- **Right Click**: Create harmonic pair from last two selected peaks
- **Double Click**: Remove a peak or pair
- **Triple Click (Left)**: Remove all peaks and pairs
- **Triple Click (Right)**: Remove all pairs but keep peaks
- **Ctrl+Click**: Remove specific peak

#### Keyboard Shortcuts
- **1** - Select highest PSD peak for all subplots
- **2** - Select 2nd highest PSD peak for all subplots
- **3** - Select 3rd highest PSD peak for all subplots
- **A** - Auto-connect all points (creates fully-connected graph)
- **L** - Remove all connecting lines (keep selected peaks)
- **C** - Clear active / selected graph
- **R** - Reset all selected peaks and lines
- **M** - Show matrix analysis
- **D** - Toggled dB ↔ Linear scale

#### Visual Elements
- **Gray dots**: Detected spectral peaks
- **Blue dots**: Selected peaks with frequency labels
- **Colorful lines**: Harmonic pairs with ratio information
- **Dashed lines**: Vertical reference lines at peak frequencies
- **Top-right legend**: Shows frequency pairs and their ratios

### Analysis Methods

1. **FFT**: Standard Fast Fourier Transform - best for general spectral analysis
2. **CQT**: Constant-Q Transform - better frequency resolution at low frequencies
3. **Multi-Res**: Multiple FFT window sizes - adaptive frequency resolution
4. **Chirplet**: Chirplet transform - good for frequency-modulated signals
5. **Chirplet Zero**: Zero-padded chirplet - enhanced for short signals
6. **Wavelet**: Wavelet packet decomposition - time-frequency localization
7. **Improved Wavelet**: Enhanced wavelet with better frequency mapping
8. **Stationary Wavelet**: Shift-invariant wavelet transform

### Output Files

The application generates several output files:

- **Interactive HTML**: Standalone visualization with full interactivity
- **Pair Data JSON**: Frequency pairs and ratios for each file/method
- **Graph Data JSON**: Node and edge data for network analysis
- **PNG Plots**: Static matplotlib figures (if saved manually)
- **CSV File**: Simple CSV with selected peaks and ratios of peak frequencies
## Configuration

### Environment Variables (Optional)

Set these in environment or modify `config.py`:

```bash
export UPLOAD_FOLDER="uploads"
export MAX_CONTENT_LENGTH= 100 * 1024 * 1024 # 100MB max file size
export NFFT=1024                    # Default FFT size
export PEAK_FMIN=100               # Default peak detection minimum (Hz)
export PEAK_FMAX=5000              # Default peak detection maximum (Hz)
export PLOT_FMIN=100               # Default plot minimum (Hz)
export PLOT_FMAX=5000              # Default plot maximum (Hz)
```

### Default Parameters

Edit these in `config.py`:

```python
DEFAULT_METHODS = ["FFT", "Chirplet Zero"]
DEFAULT_N_FFT = 1024
DEFAULT_PEAK_FMIN = 100
DEFAULT_PEAK_FMAX = 5000
MAX_PAIRS = 10
```

## API Usage

For programmatic use, import the core functions:

```python
from jellyfish_plotly_browser import compare_methods_psd_analysis, save_jellyfish_plotly

# Run analysis
fig, plots, save_func, dir_name = compare_methods_psd_analysis(
    audio_directory="path/to/audio/files",
    methods=["FFT", "CQT"],
    n_fft=1024,
    peak_fmin=100,
    peak_fmax=5000,
    use_db_scale=True
)

# Generate interactive HTML
html_path, data_path, graph_path = save_jellyfish_plotly(
    plots,
    base_filename="analysis_file",
    dir_name="favorite_animal",
    use_db_scale=True
)
```

## Troubleshooting

### Potential Issues

1. **"Python is not recognized" or "python: command not found"**
   - Install Python 3.9+ from [python.org](https://python.org)
   - On Windows: Check "Add Python to PATH" during installation
   - Restart command prompt/terminal after installation

2. **"No module named..." errors**
   - Make sure virtual environment is activated: `venv\Scripts\activate`
   - Install missing packages: `pip install -r requirements.txt`

3. **"Python version too old" error**
   - Python version is below 3.9
   - Update Python from [python.org](https://python.org)

4. **File upload fails**
   - Check file size (default limit: 100MB)
   - Ensure audio format is supported (.wav, .mp3, .flac)

5. **Browser doesn't open automatically**
   - Manually navigate to http://localhost:5000
   - Check console output for any error messages

6. **Interactive features not working in HTML**
   - Ensure JavaScript is enabled in browser
   - Try refreshing the page
   - Check browser console (F12) for JavaScript errors

### Performance Tips

- Use smaller FFT sizes (512, 1024) for faster processing
- Limit the number of analysis methods for large files
- Reduce frequency ranges for peak detection to improve speed
- Process files in smaller batches for memory efficiency

### Getting Help

1. Check that all requirements are installed: `pip list`
2. Verify Python version: `python --version` (should be 3.9+)
3. Test with smaller audio files first
4. Check browser console for JavaScript errors (F12)

## Advanced Features

### Batch Processing

Process multiple files with consistent parameters:

```python
selected_files = select_audio_files(
    directory_path="audio_samples/",
    range_start=0,
    range_end=10,  # Process first 10 files
    extensions=('.wav', '.mp3')
)
```

### Custom Peak Detection

Adjust sensitivity parameters:

```python
# More sensitive peak detection
height_percentile=0.3      # Lower threshold
prominence_factor=0.02     # Less prominence required
min_width=0.1             # Narrower peaks allowed

# Less sensitive peak detection  
height_percentile=0.8      # Higher threshold
prominence_factor=0.1      # More prominence required
min_width=2.0             # Wider peaks only
```

### Network Analysis

The tool creates network graphs of harmonic relationships. Access graph data:

```python
for plot in plots:
    graph_data = plot.get_graph_data()
    print(f"Nodes: {len(graph_data['nodes'])}")
    print(f"Edges: {len(graph_data['edges'])}")
```

## License

GPL-3.0

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly  
3. Test with smaller audio files first
4. Check browser console for JavaScript errors (F12)
5. Message me!

</small>
