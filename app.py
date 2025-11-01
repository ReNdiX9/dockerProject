from flask import Flask, render_template, send_from_directory
from pathlib import Path
import os

app = Flask(__name__)

@app.route('/outputs/figures/<filename>')
def serve_figure(filename):
    return send_from_directory('outputs/figures', filename)

@app.route('/')
def index():
    # Use existing charts from outputs/figures
    figures_dir = Path('outputs/figures')
    
    # Get the latest chart files
    charts = {
        'bar_chart': None,
        'heatmap': None,
        'scatter_plot': None,
    }
    
    if figures_dir.exists():
        # Find latest bar chart
        bar_files = list(figures_dir.glob('bar_avg_macros_*.png'))
        if bar_files:
            latest_bar = max(bar_files, key=os.path.getctime)
            charts['bar_chart'] = f'/outputs/figures/{latest_bar.name}'
        
        # Find latest heatmap
        heat_files = list(figures_dir.glob('heatmap_avg_macros_*.png'))
        if heat_files:
            latest_heat = max(heat_files, key=os.path.getctime)
            charts['heatmap'] = f'/outputs/figures/{latest_heat.name}'
        
        # Find latest scatter
        scatter_files = list(figures_dir.glob('scatter_top5_*.png'))
        if scatter_files:
            latest_scatter = max(scatter_files, key=os.path.getctime)
            charts['scatter_plot'] = f'/outputs/figures/{latest_scatter.name}'
    
    return render_template('index.html', charts=charts)

if __name__ == '__main__':
    app.run(debug=True, port=5000)