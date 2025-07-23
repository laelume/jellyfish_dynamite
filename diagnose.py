#!/usr/bin/env python3
"""
Diagnostic script for Jellyfish Dynamite Flask integration
Analyzes existing code and provides specific integration instructions
"""

# diagnose.py

import os
import sys
import ast
import re
from pathlib import Path

def find_python_files():
    """Find Python files that might contain analysis code"""
    python_files = []
    for file in os.listdir('.'):
        if file.endswith('.py') and not file.startswith('__'):
            python_files.append(file)
    return python_files

def find_html_files():
    """Find HTML files that might be working templates"""
    html_files = []
    for file in os.listdir('.'):
        if file.endswith(('.html', '.htm')):
            html_files.append(file)
        elif file.endswith('.txt'):
            # Check if txt file contains HTML
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '<!DOCTYPE html>' in content or '<html' in content:
                        html_files.append(file)
            except:
                pass
    return html_files

def analyze_python_file(filename):
    """Analyze a Python file to find relevant functions"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to find function definitions
        tree = ast.parse(content)
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        # Look for specific patterns
        analysis_functions = [f for f in functions if 'psd' in f.lower() or 'analysis' in f.lower() or 'compare' in f.lower()]
        plot_functions = [f for f in functions if 'plot' in f.lower() or 'save' in f.lower() or 'html' in f.lower()]
        interactive_classes = [c for c in classes if 'interactive' in c.lower() or 'plot' in c.lower()]
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'analysis_functions': analysis_functions,
            'plot_functions': plot_functions,
            'interactive_classes': interactive_classes,
            'total_lines': len(content.split('\n'))
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_html_file(filename):
    """Analyze HTML file to see if it's a working template"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        info = {
            'has_plotly': 'plotly' in content.lower(),
            'has_javascript': '<script' in content.lower(),
            'has_interactive': 'click' in content.lower() or 'event' in content.lower(),
            'has_your_data': 'plotData' in content or 'plot_data' in content,
            'file_size': len(content),
            'has_jellyfish': 'jellyfish' in content.lower() or 'dynamite' in content.lower()
        }
        
        # Look for template variables
        template_vars = re.findall(r'\{\{[^}]+\}\}', content)
        info['template_variables'] = template_vars
        
        return info
    except Exception as e:
        return {'error': str(e)}

def generate_integration_code(analysis_file, main_function, save_function):
    """Generate specific integration code"""
    
    integration_template = f"""
# Integration code for simple_app.py
# Replace the import section (around line 88) with:

try:
    # Import analysis functions
    from {analysis_file.replace('.py', '')} import {main_function}, {save_function}
    
    analysis_function = {main_function}
    save_function = {save_function}
    
    print("✓ Successfully imported analysis functions")
    
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print("Available functions in {analysis_file}:")
    
    # Fallback: import the whole module
    import {analysis_file.replace('.py', '')} as analysis_module
    
    # Try to find the functions dynamically
    analysis_function = getattr(analysis_module, '{main_function}', None)
    save_function = getattr(analysis_module, '{save_function}', None)
    
    if not analysis_function:
        print("❌ Could not find {main_function} function")
    if not save_function:
        print("❌ Could not find {save_function} function")

# Example usage in the Flask app:
def process_files():
    # ... file upload code ...
    
    # Call analysis function
    fig, plots, _ = analysis_function(
        audio_directory=session_dir,
        max_pairs=10,
        n_fft=n_fft,
        peak_fmin=peak_fmin,
        peak_fmax=peak_fmax,
        plot_fmin=peak_fmin,
        plot_fmax=peak_fmax,
        selected_files=saved_files,
        methods=methods,
        use_db_scale=use_db_scale
    )
    
    # Generate HTML using save function
    html_path, data_path, graph_path = save_function(
        plots, 
        base_filename=f"analysis_{{session_id}}",
        output_directory=session_dir
    )
    
    # Return the HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()
"""
    
    return integration_template

def main():
    """Main diagnostic function"""
    print("🔍 Jellyfish Dynamite - Integration Diagnostic")
    print("=" * 60)
    
    # Find relevant files
    python_files = find_python_files()
    html_files = find_html_files()
    
    print(f"\n📁 Found {len(python_files)} Python files:")
    for file in python_files:
        print(f"   • {file}")
    
    print(f"\n🌐 Found {len(html_files)} HTML files:")
    for file in html_files:
        print(f"   • {file}")
    
    # Analyze Python files
    print(f"\n🔬 Analyzing Python files...")
    analysis_candidates = []
    
    for file in python_files:
        if file in ['diagnose.py', 'simple_app.py']:
            continue
            
        print(f"\n📄 {file}:")
        info = analyze_python_file(file)
        
        if 'error' in info:
            print(f"   ❌ Error: {info['error']}")
            continue
        
        print(f"   • {len(info['functions'])} functions")
        print(f"   • {len(info['classes'])} classes") 
        print(f"   • {info['total_lines']} lines")
        
        if info['analysis_functions']:
            print(f"   🎯 Analysis functions: {', '.join(info['analysis_functions'])}")
            analysis_candidates.append((file, info['analysis_functions']))
        
        if info['plot_functions']:
            print(f"   📊 Plot functions: {', '.join(info['plot_functions'])}")
        
        if info['interactive_classes']:
            print(f"   🖱️  Interactive classes: {', '.join(info['interactive_classes'])}")
    
    # Analyze HTML files
    print(f"\n🌐 Analyzing HTML files...")
    working_html_candidates = []
    
    for file in html_files:
        print(f"\n📄 {file}:")
        info = analyze_html_file(file)
        
        if 'error' in info:
            print(f"   ❌ Error: {info['error']}")
    if __name__ == "__main__":
        main()