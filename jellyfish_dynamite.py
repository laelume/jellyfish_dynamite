# jellyfish_dynamite.py

import json
import os
import numpy as np
import webbrowser
import jelly_funcs as jelfun
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_jellyfish_template(fig, plots, base_filename="jelly_psd_template", output_directory=None):
    """Create interactive HTML using the sophisticated template."""
    
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
            output_directory = f"{daily_dir}/jelly_html"
        except:
            output_directory = "html_output"
    
    # Get timestamp
    timestamp = jelfun.get_timestamp()
    
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Determine grid dimensions
    total_plots = len(plots)
    n_cols = min(3, total_plots)
    n_rows = int(np.ceil(total_plots / n_cols))
    
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
    
    # Helper function to safely convert arrays
    def safe_tolist(arr):
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        elif isinstance(arr, (list, tuple)):
            return list(arr)
        else:
            return arr
    
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
            'frequencies': safe_tolist(plot.frequencies),
            'psd_db': safe_tolist(plot.psd_db),
            'peak_freqs': safe_tolist(plot.peak_freqs),
            'peak_powers': safe_tolist(plot.peak_powers),
            'selected_peaks': [],
            'pairs': []
        }
        plot_data.append(plot_info)
        
        # Add main PSD curve to plotly figure
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
    main_title = "Interactive PSD Analysis"
    plotly_fig.update_layout(
        title=main_title,
        height=max(600, 500 * n_rows),
        width=500 * n_cols,
        template="plotly_white",
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Get the plot data and layout as JSON strings
    #plot_data_json = json.dumps(plotly_fig.data, cls=NumpyEncoder)
    #layout_data_json = json.dumps(plotly_fig.layout, cls=NumpyEncoder)
    # Get the plot data and layout as JSON strings using Plotly's built-in serialization
    fig_json = plotly_fig.to_json()
    # Extract just the data and layout parts
    fig_dict = json.loads(fig_json)
    plot_data_json = json.dumps(fig_dict['data'])
    layout_data_json = json.dumps(fig_dict['layout'])

    plot_id = f"plot_{timestamp}"
    plot_height = max(600, 500 * n_rows)
    plot_width = 500 * n_cols
    
    # The HTML template (your exact HTML content)
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta charset="utf-8" />
    <style>
        /* Make the plot area show a grab cursor on hover */
        .js-plotly-plot .plotly .nsewdrag {
            cursor: grab !important;
        }

        /* Change to grabbing cursor when actively dragging */
        .js-plotly-plot .plotly .nsewdrag.dragging {
            cursor: grabbing !important;
        }

        /* Also apply grabbing cursor when mouse is down */
        .js-plotly-plot .plotly .nsewdrag:active {
            cursor: grabbing !important;
        }

        /* Use JavaScript to add mouse down/up events */
        .plotly-graph-div {
            cursor: auto;
        }

        /* Additional styling for hover effects */
        .js-plotly-plot .plotly .cursor-pointer {
            cursor: pointer !important;
        }

        /* Improve axis labels */
        .js-plotly-plot .plotly .xtitle, .js-plotly-plot .plotly .ytitle {
            font-weight: bold !important;
        }

        /* Container styling */
        .instructions {
            max-width: 1200px;
            margin: 20px auto;
            background-color: #f5f8fa;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .instructions h2 {
            color: #2c3e50;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }

        .instructions .controls {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .instructions .section {
            flex: 1;
            min-width: 300px;
        }

        .instructions h3 {
            color: #3498db;
        }

        .instructions ul {
            list-style-type: none;
            padding-left: 10px;
        }

        .instructions .highlight {
            font-weight: bold;
            color: #e74c3c;
        }

        /* Mobile-specific adjustments */
        @media (max-width: 768px) {
            .instructions {
                margin: 10px;
                padding: 10px;
            }

            .instructions .section {
                min-width: 100%;
            }

            .js-plotly-plot {
                height: auto !important;
                width: 100% !important;
            }

            .plotly .main-svg {
                width: 100% !important;
            }

            .modebar-container {
                transform: scale(0.8);
                transform-origin: top right;
            }
        }

        /* Improved CSS for better dark/light mode tooltip contrast */

        /* Base styling for all hover tooltips */
        .js-plotly-plot .plotly .hoverlayer .hover,
        .js-plotly-plot .plotly .annotation-text {
            transition: background-color 0.2s ease, color 0.2s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            border-radius: 4px !important;
        }

        /* Light mode styling (default) */
        .js-plotly-plot .plotly .hoverlayer .hover,
        .js-plotly-plot .plotly .annotation-text {
            background-color: rgba(240, 240, 240, 0.9) !important;
            color: #222222 !important;
            border: 1px solid #dddddd !important;
        }

        .js-plotly-plot .plotly .hoverlayer .hover text,
        .js-plotly-plot .plotly .annotation-text text {
            fill: #222222 !important;
        }

        /* Dark mode styling using prefers-color-scheme */
        @media (prefers-color-scheme: dark) {
            /* Base theme adjustments */
            body {
                background-color: #121212 !important;
                color: #e0e0e0 !important;
            }

            /* Tooltip styling for dark mode */
            .js-plotly-plot .plotly .hoverlayer .hover,
            .js-plotly-plot .plotly .annotation-text {
                background-color: rgba(70, 70, 70, 0.9) !important;
                color: #e0e0e0 !important;
                border: 1px solid #555555 !important;
            }

            .js-plotly-plot .plotly .hoverlayer .hover text,
            .js-plotly-plot .plotly .annotation-text text {
                fill: #e0e0e0 !important;
            }

            /* Ensure hover text is clearly visible */
            .js-plotly-plot .plotly .hoverlayer .hover text[data-unformatted],
            .js-plotly-plot .plotly .annotation-text text {
                fill: #e0e0e0 !important;
                font-weight: 500 !important;
                text-shadow: 0 1px 1px rgba(0, 0, 0, 0.3) !important;
            }

            /* Different styling for peak points vs cursor position */
            .js-plotly-plot .plotly .annotation[data-index="peak-point"] {
                background-color: rgba(80, 80, 80, 0.8) !important;
            }

            .js-plotly-plot .plotly .annotation[data-index="cursor-point"] {
                background-color: rgba(60, 60, 60, 0.8) !important;
            }

            /* Improve main plot background for dark mode */
            .js-plotly-plot .plotly .plot-container {
                background-color: #222222 !important;
            }

            /* Better contrast for the main PSD line */
            .js-plotly-plot .plotly .scatter .lines path {
                stroke: rgba(255, 255, 255, 0.8) !important; 
                stroke-width: 2px !important;
            }

            /* Better instructions styling for dark mode */
            .instructions {
                background-color: #2a2a2a !important;
                color: #e0e0e0 !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            }

            .instructions h2 {
                color: #e0e0e0 !important;
                border-bottom: 1px solid #444 !important;
            }

            .instructions h3 {
                color: #60a5fa !important;
            }

            .instructions .highlight {
                color: #f87171 !important;
            }

            /* Adjust modebar for dark mode */
            .modebar {
                background-color: rgba(50, 50, 50, 0.7) !important;
            }

            .modebar-btn path {
                fill: #e0e0e0 !important;
            }
        }

        /* Style differences between peak point and cursor annotations */
        .js-plotly-plot .plotly .annotation[data-index="peak-point"] {
            font-weight: bold !important;
        }

        .js-plotly-plot .plotly .annotation[data-index="cursor-point"] {
            font-weight: normal !important;
        }

        /* Additional styling for readability in dark mode */
        @media (prefers-color-scheme: dark) {
            /* Improve axis color and readability */
            .js-plotly-plot .plotly .xaxis .xtick text,
            .js-plotly-plot .plotly .yaxis .ytick text,
            .js-plotly-plot .plotly .xtitle, 
            .js-plotly-plot .plotly .ytitle {
                fill: #e0e0e0 !important;
            }

            /* Improve plot title readability */
            .js-plotly-plot .plotly .gtitle {
                fill: #ffffff !important;
            }

            /* Make sure grid lines are visible but subtle */
            .js-plotly-plot .plotly .gridlayer path {
                stroke: rgba(150, 150, 150, 0.2) !important;
            }

            /* Improve subplot visibility */
            .js-plotly-plot .plotly .subplot {
                background-color: rgba(34, 34, 34, 0.9) !important;
            }
        }

        /* Make sure the text always has good contrast against its background */
        .js-plotly-plot .plotly text {
            font-family: 'Arial', sans-serif !important;
        }
    </style>
</head>
<body>
    <div class="instructions">
        <h2>Interactive PSD Analysis</h2>
        <div class="controls">
            <div class="section">
                <h3>Navigation Controls:</h3>
                <ul>
                    <li><span class="highlight">✋ Click and drag</span> to grab and pan the view (like Google Maps)</li>
                    <li><span class="highlight">⚙️ Use the scroll wheel</span> to zoom in and out</li>
                    <li><span class="highlight">🔍 Double-click</span> to reset the view</li>
                </ul>
            </div>
            <div class="section">
                <h3>Tools (toolbar at top-right):</h3>
                <ul>
                    <li><span class="highlight">📸 Camera icon</span> to download as PNG</li>
                    <li><span class="highlight">✏️ Draw line</span> for measuring frequencies</li>
                    <li><span class="highlight">🧹 Eraser</span> to remove drawn lines</li>
                </ul>
            </div>
        </div>
        <div class="mobile-note">
            <p><strong>Note for mobile users:</strong> Use pinch gestures to zoom and drag with single finger to pan. Tap the menu icon to access tools.</p>
        </div>
    </div>
    
    <!-- Dynamic plot container -->
    <div>
        <script type="text/javascript">window.PlotlyConfig = {{MathJaxConfig: 'local'}};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.0.min.js"></script>
        <div id="{plot_id}" class="plotly-graph-div" style="height:{plot_height}px; width:{plot_width}px;"></div>
        <script type="text/javascript">
            window.PLOTLYENV = window.PLOTLYENV || {{}};
            if (document.getElementById("{plot_id}")) {{
                Plotly.newPlot(
                    "{plot_id}",
                    {plot_data},
                    {layout_data},
                    {{
                        "responsive": true, 
                        "displayModeBar": true, 
                        "scrollZoom": true, 
                        "modeBarButtonsToAdd": ["drawline", "eraseshape"], 
                        "displaylogo": false, 
                        "editable": false, 
                        "staticPlot": false
                    }}
                );
            }}
        </script>
    </div>

    <!-- Custom drag handling script -->
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        // Wait for Plotly to initialize
        setTimeout(function() {{
            const graphDivs = document.querySelectorAll('.js-plotly-plot');

            graphDivs.forEach(function(div) {{
                let isDragging = false;
                let startX, startY;
                let originalViewport = null;

                // Get the Plotly graph object
                const gd = div._fullLayout ? div : div.parentNode;

                // Override default drag behavior
                div.addEventListener('mousedown', function(e) {{
                    // Only handle primary button (left-click)
                    if (e.button !== 0) return;

                    // Skip if clicking on a legend or a plotly control
                    if (e.target.closest('.legend') || e.target.closest('.modebar')) return;

                    // Set dragging state
                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;

                    // Save current viewport
                    try {{
                        originalViewport = {{
                            xaxis: Object.assign({{}}, gd._fullLayout.xaxis),
                            yaxis: Object.assign({{}}, gd._fullLayout.yaxis)
                        }};
                    }} catch(err) {{
                        console.log('Error getting viewport:', err);
                        return;
                    }}

                    // Change cursor to grabbing
                    document.body.style.cursor = 'grabbing';
                    div.style.cursor = 'grabbing';

                    // Prevent default behavior
                    e.preventDefault();
                }});

                // Handle drag motion
                document.addEventListener('mousemove', function(e) {{
                    if (!isDragging || !originalViewport) return;

                    // Calculate distance moved
                    const deltaX = e.clientX - startX;
                    const deltaY = e.clientY - startY;

                    try {{
                        // Convert pixel distance to data coordinates
                        const xRange = originalViewport.xaxis.range;
                        const yRange = originalViewport.yaxis.range;
                        const layout = gd._fullLayout;

                        const pixelToDataRatioX = (xRange[1] - xRange[0]) / layout.width;
                        const pixelToDataRatioY = (yRange[1] - yRange[0]) / layout.height;

                        // Calculate new ranges
                        const newXRange = [
                            xRange[0] - deltaX * pixelToDataRatioX,
                            xRange[1] - deltaX * pixelToDataRatioX
                        ];

                        const newYRange = [
                            yRange[0] + deltaY * pixelToDataRatioY,
                            yRange[1] + deltaY * pixelToDataRatioY
                        ];

                        // Update the plot ranges
                        Plotly.relayout(gd, {{
                            'xaxis.range': newXRange,
                            'yaxis.range': newYRange
                        }});
                    }} catch(err) {{
                        console.log('Error during drag:', err);
                    }}
                }});

                // End dragging
                document.addEventListener('mouseup', function() {{
                    if (isDragging) {{
                        isDragging = false;
                        document.body.style.cursor = '';
                        div.style.cursor = '';
                    }}
                }});

                // Handle mouse leave
                document.addEventListener('mouseleave', function() {{
                    if (isDragging) {{
                        isDragging = false;
                        document.body.style.cursor = '';
                        div.style.cursor = '';
                    }}
                }});

                // Prevent default behavior of plotly drag which conflicts
                div.addEventListener('plotly_relayouting', function(e) {{
                    if (isDragging) {{
                        // This prevents default plotly drag
                        e.preventDefault();
                    }}
                }});
            }});
        }}, 1000); // Wait 1 second for Plotly to initialize
    }});
    </script>
</body>
</html>"""
    
    # Replace template placeholders
    html_str = html_template.format(
        plot_id=plot_id,
        plot_height=plot_height,
        plot_width=plot_width,
        plot_data=plot_data_json,
        layout_data=layout_data_json
    )
    
    # Define file paths
    html_filename = f"{base_filename}_{timestamp}.html"
    html_path = os.path.join(output_directory, html_filename)
    
    # Save HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    
    # Save data files
    data_filename = f"{base_filename}_{timestamp}_pairdata.json"
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
    
    # Save graph data
    graph_filename = f"{base_filename}_{timestamp}_graphdata.json"
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
    
    print(f"HTML saved to: {html_path}")
    print(f"Data saved to: {data_path}")
    print(f"Graph data saved to: {graph_path}")
    
    try:
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
        print(f"Opening HTML file in browser...")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open manually: {html_path}")
    
    return html_path, data_path, graph_path