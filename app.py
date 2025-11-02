from flask import Flask, render_template, send_from_directory, jsonify, request
from pathlib import Path
import os
import pandas as pd
import subprocess
from sklearn.cluster import KMeans
import numpy as np
import math

app = Flask(__name__)

@app.route('/outputs/figures/<filename>')
def serve_figure(filename):
    return send_from_directory('outputs/figures', filename)

@app.route('/')
def index():
    figures_dir = Path('outputs/figures')

    page = request.args.get('page', 1, type=int)
    total_pages = 4
    prev_page = page - 1 if page > 1 else 1
    next_page = page + 1 if page < total_pages else total_pages

    # Decide which template to render
    if page == 1:
        template_name = 'index.html'
    elif page == 2:
        template_name = 'nutritional.html'
    elif page == 3:
        template_name = 'recipes.html'
    elif page == 4:
        template_name = 'clusters.html'
    else:
        template_name = 'index.html'

    # Charts code (only relevant for index.html or can include in all pages)
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

    # Finally render the chosen template with charts and pagination
    return render_template(
        template_name,
        charts=charts,
        page=page,
        prev_page=prev_page,
        next_page=next_page,
        total_pages=total_pages
    )


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
         .str.title()
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

@app.route('/nutritional')
def nutritional():
    return render_template('nutritional.html')

@app.route('/api/nutritional-insight')
def api_nutritional_insight():
    """
    Return average Protein, Carbs, Fat grouped by Diet_type or Cuisine_type.
    Query param: group=Diet_type or group=Cuisine_type
    """
    csv_path = Path('All_Diets.csv')
    if not csv_path.exists():
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(csv_path)
    group_by = request.args.get('group', 'Diet_type')

    if group_by not in df.columns:
        return jsonify({"error": f"Invalid group: {group_by}"}), 400

    # Calculate averages
    avg_df = df.groupby(group_by)[['Protein(g)', 'Carbs(g)', 'Fat(g)']].mean().round(2)
    result = avg_df.to_dict(orient='index')
    return jsonify(result)

@app.route('/recipes')
def recipes():
    return render_template('recipes.html')

@app.route('/api/recipes')
def api_recipes():
    """
    Return recipes filtered by diet and/or cuisine.
    Query params: diet=<diet_name>, cuisine=<cuisine_name>
    """
    csv_path = Path('All_Diets.csv')
    if not csv_path.exists():
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(csv_path)
    diet = request.args.get('diet', '').strip().title()
    cuisine = request.args.get('cuisine', '').strip().title()

    if diet:
        df = df[df['Diet_type'].str.title() == diet]
    if cuisine:
        df = df[df['Cuisine_type'].str.title() == cuisine]

    df = df[['Recipe_name', 'Diet_type', 'Cuisine_type', 'Protein(g)', 'Carbs(g)', 'Fat(g)']]
    return jsonify(df.to_dict(orient='records'))

@app.route('/clusters')
def clusters():
    return render_template('clusters.html')

#if charts are already in outputs/figures/  run without subprocess part (win-python app.py linux python3 app.py)
#or just do - sudo apt install python-is-python3 
if __name__ == '__main__':
    # Automatically run your data analysis script
    csv_path = 'All_Diets.csv'  # adjust if it's in another folder
    try:
        print("Running data_analysis.py before starting Flask...")
        subprocess.run(
            ['python', 'data_analysis.py', csv_path],
            check=True
        )
        print("Data analysis completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Data analysis script failed:", e)

    # Now start Flask
    app.run(debug=True, port=5000)

