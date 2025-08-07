#!/usr/bin/env python3

"""
Jellyfish Dynamite Flask App
Interactive PSD Analysis Tool
"""
import os
import sys
import webbrowser
import threading

# macOS threading fixes - MUST be before other imports
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


# Check Python version with platform-specific help
if sys.version_info < (3, 9):
    print("❌ Error: Python 3.9 or higher is required")
    print(f"   You are using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()
    if platform.system() == "Darwin":  # Mac
        print("   Install from: https://python.org")
        print("   Or use Homebrew: brew install python3")
    elif platform.system() == "Windows":
        print("   Install from: https://python.org")
        print("   Make sure to check 'Add Python to PATH'")
    else:  # Linux
        print("   Install using package manager or from python.org")
    sys.exit(1)

print(f"✅ Running on Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


# jelly_app.py

from flask import Flask, request, render_template, jsonify, send_file, send_from_directory, abort
import uuid
import tempfile
import shutil
from pathlib import Path
import time

from config import Config
from analysis_service import AnalysisService

app = Flask(__name__)
app.config.from_object(Config)

# Initialize services
analysis_service = AnalysisService(app.config)


@app.route('/')
def upload_form():
    """Render upload form with dynamic configuration"""
    return render_template('upload_form.html', 
                         config=app.config,
                         methods=Config.DEFAULT_METHODS)


@app.route('/process', methods=['POST'])
def process_files():
    """Streamlined processing with better error handling"""
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    
    try:
        os.makedirs(session_dir, exist_ok=True)
        
        # Validate and save files
        files = request.files.getlist('files')
        valid_files = analysis_service.validate_files(files)
        
        if not valid_files:
            return jsonify({'error': 'No valid audio files uploaded'}), 400
        
        # Save files
        saved_files = []
        for file in valid_files:
            filepath = os.path.join(session_dir, file.filename)
            file.save(filepath)
            saved_files.append(file)
        
        # Parse parameters - UPDATED with resolution options
        psd_n_fft = int(request.form.get('psd_n_fft', Config.DEFAULT_N_FFT))
        spec_n_fft = int(request.form.get('spec_n_fft', Config.DEFAULT_SPEC_N_FFT))
        hop_ratio = int(request.form.get('hop_ratio', Config.DEFAULT_HOP_RATIO))
        
        # Calculate hop lengths
        psd_hop_length = psd_n_fft // hop_ratio
        spec_hop_length = spec_n_fft // hop_ratio
        
        params = {
            'methods': request.form.getlist('methods') or Config.DEFAULT_METHODS,
            'psd_n_fft': psd_n_fft, 
            'spec_n_fft': spec_n_fft, 
            'psd_hop_length': psd_hop_length, 
            'spec_hop_length': spec_hop_length, 
            'n_fft': psd_n_fft,  # Keep for backward compatibility
            'peak_fmin': int(request.form.get('peak_fmin', os.environ.get('PEAK_FMIN', Config.DEFAULT_PEAK_FMIN))),
            'peak_fmax': int(request.form.get('peak_fmax', os.environ.get('PEAK_MAX', Config.DEFAULT_PEAK_FMAX))),
            'plot_fmin': int(request.form.get('plot_fmin', os.environ.get('PLOT_FMIN', Config.DEFAULT_PEAK_FMIN))),
            'plot_fmax': int(request.form.get('plot_fmax', os.environ.get('PLOT_MAX', Config.DEFAULT_PEAK_FMAX))),
            'use_db_scale': 'use_db_scale' in request.form, 
            'dir_name': request.form.get('dir_name', '').strip() or 'analysis'
        }
        
        # Run analysis
        result = analysis_service.process_analysis(session_dir, saved_files, params)
        
        if result['success']:
            # Load and modify HTML
            with open(result['html_path'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Add navigation and timing info - UPDATED with resolution display
            html_content = add_flask_navigation(html_content, result['processing_time'], params)
            return html_content
        
        else:
            return render_template('error.html', 
                                 error=result['error'],
                                 session_id=session_id), 500
    
    except Exception as e:
        return render_template('error.html', 
                             error=str(e),
                             session_id=session_id), 500

def add_flask_navigation(html_content, processing_time, params):
    """Add Flask-specific UI elements to generated HTML"""
    # UPDATED - Add resolution info display
    nav_html = f'''
    <div style="position: fixed; top: 10px; left: 10px; z-index: 2000; background: rgba(0,123,255,0.9); padding: 15px; border-radius: 5px; color: white; font-family: monospace; font-size: 12px;">
        <a href="/" style="color: white; text-decoration: none; font-weight: bold;">← New Analysis</a>
        <div style="border-top: 1px solid rgba(255,255,255,0.3); margin: 8px 0; padding-top: 8px;">
            <strong>Resolution Settings:</strong><br>
            PSD N_FFT: {params.get('psd_n_fft', 'N/A')}<br>
            Spec N_FFT: {params.get('spec_n_fft', 'N/A')}<br>
            PSD Hop: {params.get('psd_hop_length', 'N/A')}<br>
            Spec Hop: {params.get('spec_hop_length', 'N/A')}
        </div>
        <div style="color: rgba(255,255,255,0.7); font-size: 11px; margin-top: 8px;">
            Generated in {processing_time:.2f}s
        </div>
    </div>
    '''
    
    return html_content.replace('<body>', f'<body>{nav_html}')


# Error handlers and cleanup routes
@app.errorhandler(413)
def too_large(e):
    return "File too large", 413

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/tranche/<path:filename>')
def serve_audio(filename):
    """Serve audio files for the synthesizer"""
    try:
        return send_from_directory('tranche', filename)
    except FileNotFoundError:
        abort(404)

@app.route('/<path:filename>')
def serve_root_files(filename):
    """Serve files from root directory"""
    try:
        return send_from_directory('.', filename)
    except FileNotFoundError:
        abort(404)

@app.route('/api/audio/<path:filename>')
def api_serve_audio(filename):
    """API endpoint for audio file access with CORS headers"""
    response = send_from_directory('tranche', filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    return response

@app.route('/api/analysis/status/<analysis_id>')
def get_analysis_status(analysis_id):
    """Get status of running analysis"""
    # Implementation for async analysis tracking
    return jsonify({'status': 'running', 'progress': 50})

@app.route('/api/peaks', methods=['POST'])
def save_peaks():
    """Save peak selections from the frontend"""
    data = request.json
    # Save peak data to database or file
    return jsonify({'success': True, 'message': 'Peaks saved'})

@app.route('/api/audio/synthesize', methods=['POST'])
def synthesize_audio():
    """Generate sine waves for selected peaks"""
    data = request.json
    frequencies = data.get('frequencies', [])
    # Implementation for audio synthesis
    return jsonify({'success': True, 'audio_url': '/generated_audio.wav'})

@app.route('/temp_uploads/<session_id>/<filename>')
def serve_session_audio(session_id, filename):
    """Serve audio files from session directories"""
    print(f"🎵 Route called: session_id={session_id}, filename={filename}")
    
    try:
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        full_path = os.path.join(session_path, filename)
        
        print(f"🎵 Looking for file at: {full_path}")
        print(f"🎵 File exists: {os.path.exists(full_path)}")
        
        if os.path.exists(full_path):
            print(f"🎵 Files in directory: {os.listdir(session_path)}")
        
        response = send_from_directory(session_path, filename)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET'
        return response
    except Exception as e:
        print(f"❌ Error serving audio: {e}")
        abort(404)


if __name__ == '__main__':
    # Setup
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("🎐 Jellyfish Dynamite Flask App 🎐")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("🌐 Server: http://localhost:5000")
    
    # Auto-open browser (only in main process, not debug reloader)
    def open_browser():
        webbrowser.open('http://localhost:5000')

    # Only open browser if not in reloader subprocess
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        threading.Timer(1.5, open_browser).start()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)