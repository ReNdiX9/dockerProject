from flask import Flask, render_template, send_from_directory, jsonify, request
from pathlib import Path
import os
import pandas as pd
import subprocess
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/outputs/figures/<filename>')
def serve_figure(filename):
    return send_from_directory('outputs/figures', filename)

def get_latest_chart(figures_dir: Path, pattern: str) -> str:
    """Get the latest HTML chart file matching the pattern"""
    files = list(figures_dir.glob(pattern))
    if files:
        latest = max(files, key=os.path.getctime)
        # Read the HTML content
        with open(latest, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract just the div content (remove the full HTML wrapper if present)
        # Plotly writes a full HTML document, we want just the chart div
        if '<body>' in content:
            # Extract content between body tags
            start = content.find('<body>')
            end = content.find('</body>')
            if start != -1 and end != -1:
                content = content[start+6:end].strip()
        return content
    return None

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

    # Get interactive charts HTML content
    charts = {
        'bar_chart': None,
        'heatmap': None,
        'scatter_plot': None,
        'pie_chart': None,
        'cluster_chart': None,
    }

    if figures_dir.exists():
        charts['bar_chart'] = get_latest_chart(figures_dir, 'bar_avg_macros_*.html')
        charts['heatmap'] = get_latest_chart(figures_dir, 'heatmap_avg_macros_*.html')
        charts['scatter_plot'] = get_latest_chart(figures_dir, 'scatter_top5_*.html')
        charts['pie_chart'] = get_latest_chart(figures_dir, 'pie_recipe_distribution_*.html')
        charts['cluster_chart'] = get_latest_chart(figures_dir, 'clusters_*.html')

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

@app.route('/api/clusters')
def api_get_clusters():
    """Return cluster information and recipes by cluster"""
    csv_path = Path('All_Diets.csv')
    if not csv_path.exists():
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(csv_path)
    
    # Perform clustering
    features = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Format the response
    clusters_data = []
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        common_diet = cluster_data['Diet_type'].mode()
        common_cuisine = cluster_data['Cuisine_type'].mode()
        
        clusters_data.append({
            'cluster': int(cluster),
            'recipe_count': len(cluster_data),
            'avg_protein': float(cluster_data['Protein(g)'].mean()),
            'avg_carbs': float(cluster_data['Carbs(g)'].mean()),
            'avg_fat': float(cluster_data['Fat(g)'].mean()),
            'common_diet': common_diet.iloc[0] if not common_diet.empty else 'Unknown',
            'common_cuisine': common_cuisine.iloc[0] if not common_cuisine.empty else 'Unknown',
            'description': generate_cluster_description(cluster_data)
        })
    
    return jsonify({
        'clusters': clusters_data,
        'total_clusters': len(clusters_data)
    })

def generate_cluster_description(cluster_data):
    """Generate a human-readable description of the cluster"""
    avg_protein = cluster_data['Protein(g)'].mean()
    avg_carbs = cluster_data['Carbs(g)'].mean()
    avg_fat = cluster_data['Fat(g)'].mean()
    
    if avg_protein > avg_carbs and avg_protein > avg_fat:
        return "High Protein Cluster"
    elif avg_carbs > avg_protein and avg_carbs > avg_fat:
        return "High Carb Cluster"
    elif avg_fat > avg_protein and avg_fat > avg_carbs:
        return "High Fat Cluster"
    else:
        return "Balanced Macronutrient Cluster"

@app.route('/api/cluster-recipes/<int:cluster_id>')
def api_get_cluster_recipes(cluster_id):
    """Return recipes for a specific cluster"""
    csv_path = Path('All_Diets.csv')
    if not csv_path.exists():
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(csv_path)
    
    # Perform clustering
    features = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    cluster_recipes = df[df['Cluster'] == cluster_id][
        ['Recipe_name', 'Diet_type', 'Cuisine_type', 'Protein(g)', 'Carbs(g)', 'Fat(g)']
    ].to_dict('records')
    
    return jsonify(cluster_recipes)

if __name__ == '__main__':
    # Automatically run your data analysis script
    csv_path = 'All_Diets.csv'
    try:
        print("Running data_analysis.py to generate interactive charts...")
        subprocess.run(
            ['python', 'data_analysis.py', csv_path],
            check=True
        )
        print("Interactive charts generated successfully.")
    except subprocess.CalledProcessError as e:
        print("Data analysis script failed:", e)

    # Now start Flask
    app.run(debug=True, port=5000)