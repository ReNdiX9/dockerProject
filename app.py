from flask import Flask, render_template, send_from_directory, jsonify, request
from pathlib import Path
import os
import pandas as pd

app = Flask(__name__)

@app.route('/outputs/figures/<filename>')
def serve_figure(filename):
    return send_from_directory('outputs/figures', filename)

@app.route('/')
def index():
    figures_dir = Path('outputs/figures')

    charts = {
        'bar_chart': None,
        'heatmap': None,
        'scatter_plot': None,
        'pie_chart': None,
    }

    if figures_dir.exists():
        # Latest bar
        bar_files = list(figures_dir.glob('bar_avg_macros_*.png'))
        if bar_files:
            latest_bar = max(bar_files, key=os.path.getctime)
            ts = int(os.path.getmtime(latest_bar))
            charts['bar_chart'] = f'/outputs/figures/{latest_bar.name}?t={ts}'

        # Latest heatmap
        heat_files = list(figures_dir.glob('heatmap_avg_macros_*.png'))
        if heat_files:
            latest_heat = max(heat_files, key=os.path.getctime)
            ts = int(os.path.getmtime(latest_heat))
            charts['heatmap'] = f'/outputs/figures/{latest_heat.name}?t={ts}'

        # Latest scatter
        scatter_files = list(figures_dir.glob('scatter_top5_*.png'))
        if scatter_files:
            latest_scatter = max(scatter_files, key=os.path.getctime)
            ts = int(os.path.getmtime(latest_scatter))
            charts['scatter_plot'] = f'/outputs/figures/{latest_scatter.name}?t={ts}'

        # Latest pie
        pie_files = list(figures_dir.glob('pie_recipe_distribution_*.png'))
        if pie_files:
            latest_pie = max(pie_files, key=os.path.getctime)
            ts = int(os.path.getmtime(latest_pie))
            charts['pie_chart'] = f'/outputs/figures/{latest_pie.name}?t={ts}'

    return render_template('index.html', charts=charts)

@app.route('/api/diets')
def api_get_diets():
    """Return normalized, unique diet types from the raw CSV."""
    csv_path = Path('All_Diets.csv')
    if not csv_path.exists():
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(csv_path)
    s = df.get("Diet_type")
    if s is None:
        return jsonify([])

    diets = (
        s.dropna()
         .astype(str)
         .str.strip()
         .str.title()   # match normalization in analysis script
         .unique()
         .tolist()
    )
    return jsonify(sorted(diets))

@app.route('/api/avg-macros')
def api_get_avg_macros():
    """Return averages from the latest generated avg_macros CSV."""
    tables_dir = Path('outputs/tables')
    files = sorted(tables_dir.glob('avg_macros_by_diet_*.csv'))
    if not files:
        return jsonify({"error": "No avg_macros table found. Run data_analysis.py first."}), 404

    latest = files[-1]
    df = pd.read_csv(latest, index_col=0)

    diet = request.args.get('diet', 'all')
    if diet != 'all':
        if diet in df.index:
            result = df.loc[[diet]].round(2).to_dict(orient="index")
        else:
            result = {}
    else:
        result = df.round(2).to_dict(orient="index")

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
