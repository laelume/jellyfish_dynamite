# analysis_service.py

import jellyfish_plotly_browser as jelbrow
import jelly_funcs as jelfun
import os
import time
from pathlib import Path

class AnalysisService:
    def __init__(self, config):
        self.config = config
    
    def validate_files(self, files):
        """Validate uploaded files"""
        valid_extensions = {'.wav', '.mp3', '.flac'}
        valid_files = []
        
        for file in files:
            if file.filename:
                ext = Path(file.filename).suffix.lower()
                if ext in valid_extensions:
                    valid_files.append(file)
                else:
                    print(f"Skipping invalid file: {file.filename}")
        
        return valid_files
    

    
    def process_analysis(self, session_dir, files, params):
        """Main analysis processing"""
        start_time = time.time()
        
        try:
            # Remove dir_name from params for the analysis function only
            analysis_params = {k: v for k, v in params.items() if k != 'dir_name'}

            # # Separate analysis params from HTML generation params
            # analysis_params = {
            #     k: v for k, v in params.items() 
            #     if k not in ['dir_name', 'use_db_scale']
            # }

            # Run analysis
            fig, plots, _, dir_short_name = jelbrow.compare_methods_psd_analysis(
                audio_directory=session_dir,
                max_pairs=10,
                selected_files=[f.filename for f in files],
                **analysis_params # Use filtered params
            )
            
            # Generate HTML
            result = jelbrow.save_jellyfish_plotly(
                plots,
                base_filename=f"analysis_{int(time.time())}",
                output_directory=session_dir,
                #dir_name=dir_short_name, # Changed to use the user-provided name from params
                **params
            )
            
            processing_time = time.time() - start_time
            return {
                'success': True,
                'html_path': result[0] if isinstance(result, tuple) else result,
                'processing_time': processing_time,
                'files_processed': len(files)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }