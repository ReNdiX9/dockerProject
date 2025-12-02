#!/usr/bin/env python3
"""
data_analysis.py — Task 1 automation with interactive Plotly charts

What this script does
---------------------
1) Loads the All_Diets.csv dataset.
2) Cleans and normalizes key columns (diet/cuisine names; numeric macros).
3) Fills missing numeric values with the column mean.
4) Adds engineered metrics:
   - Protein_to_Carbs_ratio
   - Carbs_to_Fat_ratio
5) Computes required insights and generates interactive Plotly charts
6) Saves charts as HTML files that can be embedded in templates

Usage
-----
python data_analysis.py /path/to/All_Diets.csv --out outputs
"""

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = [
    "Diet_type",
    "Recipe_name",
    "Cuisine_type",
    "Protein(g)",
    "Carbs(g)",
    "Fat(g)",
]


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def read_and_normalize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    col_map = {
        "Protein (g)": "Protein(g)",
        "Carbs (g)": "Carbs(g)",
        "Fat (g)": "Fat(g)",
        "Cuisine": "Cuisine_type",
        "Cuisine Type": "Cuisine_type",
        "Diet": "Diet_type",
        "Diet Type": "Diet_type",
        "Recipe": "Recipe_name",
        "Recipe Name": "Recipe_name",
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for text_col in ["Diet_type", "Cuisine_type"]:
        df[text_col] = (
            df[text_col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "none": np.nan})
        )
        df[text_col] = df[text_col].apply(lambda x: x.title() if isinstance(x, str) else x)

    for num_col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    for num_col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        mean_val = df[num_col].mean(skipna=True)
        df[num_col] = df[num_col].fillna(mean_val)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    carbs = df["Carbs(g)"].to_numpy()
    fat = df["Fat(g)"].to_numpy()
    protein = df["Protein(g)"].to_numpy()

    prot_to_carbs = np.where(carbs != 0, protein / carbs, np.nan)
    carbs_to_fat = np.where(fat != 0, carbs / fat, np.nan)

    df["Protein_to_Carbs_ratio"] = prot_to_carbs
    df["Carbs_to_Fat_ratio"] = carbs_to_fat
    return df


def compute_insights(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    avg_macros = (
        df.groupby("Diet_type", as_index=True)[["Protein(g)", "Carbs(g)", "Fat(g)"]]
        .mean()
        .sort_index()
    )

    top5_protein = (
        df.sort_values(["Diet_type", "Protein(g)"], ascending=[True, False])
        .groupby("Diet_type", group_keys=False)
        .head(5)
        .loc[:, ["Diet_type", "Recipe_name", "Cuisine_type", "Protein(g)", "Carbs(g)", "Fat(g)"]]
    )

    highest_protein_by_diet = (
        avg_macros["Protein(g)"]
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Protein(g)": "Mean_Protein(g)"})
    )

    cuisine_counts = (
        df.groupby(["Diet_type", "Cuisine_type"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Diet_type", "Count"], ascending=[True, False])
    )

    return avg_macros, top5_protein, highest_protein_by_diet, cuisine_counts


def save_tables(out_dir: Path, avg_macros, top5, highest, cuisines) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    avg_macros.to_csv(tables_dir / f"avg_macros_by_diet_{ts}.csv")
    top5.to_csv(tables_dir / f"top5_protein_by_diet_{ts}.csv", index=False)
    highest.to_csv(tables_dir / f"diet_highest_mean_protein_{ts}.csv", index=False)
    cuisines.to_csv(tables_dir / f"cuisine_counts_by_diet_{ts}.csv", index=False)


def create_bar_chart(avg_macros: pd.DataFrame, fig_dir: Path) -> Path:
    """Create interactive bar chart with Plotly"""
    diets = avg_macros.index.to_list()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Protein(g)',
        x=diets,
        y=avg_macros['Protein(g)'],
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        name='Carbs(g)',
        x=diets,
        y=avg_macros['Carbs(g)'],
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.add_trace(go.Bar(
        name='Fat(g)',
        x=diets,
        y=avg_macros['Fat(g)'],
        marker_color='rgb(50, 171, 96)'
    ))
    
    fig.update_layout(
        title='Average Macronutrients by Diet Type',
        xaxis_title='Diet Type',
        yaxis_title='Grams (g)',
        barmode='group',
        xaxis_tickangle=-45,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    out_path = fig_dir / f"bar_avg_macros_{timestamp()}.html"
    fig.write_html(str(out_path), include_plotlyjs='cdn', div_id='bar-chart')
    return out_path


def create_heatmap(avg_macros: pd.DataFrame, fig_dir: Path) -> Path:
    """Create interactive heatmap with Plotly"""
    data = avg_macros[["Protein(g)", "Carbs(g)", "Fat(g)"]].to_numpy()
    diets = avg_macros.index.to_list()
    nutrients = ["Protein(g)", "Carbs(g)", "Fat(g)"]
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=nutrients,
        y=diets,
        colorscale='Viridis',
        text=np.round(data, 1),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Diet: %{y}<br>Nutrient: %{x}<br>Value: %{z:.1f}g<extra></extra>'
    ))
    
    fig.update_layout(
        title='Heatmap: Average Macros by Diet Type',
        xaxis_title='Macronutrients',
        yaxis_title='Diet Type',
        template='plotly_white',
        height=600
    )
    
    out_path = fig_dir / f"heatmap_avg_macros_{timestamp()}.html"
    fig.write_html(str(out_path), include_plotlyjs='cdn', div_id='heatmap')
    return out_path


def create_scatter_plot(top5: pd.DataFrame, fig_dir: Path) -> Path:
    """Create interactive scatter plot with Plotly"""
    fig = px.scatter(
        top5,
        x='Carbs(g)',
        y='Protein(g)',
        size='Fat(g)',
        color='Diet_type',
        hover_data=['Recipe_name', 'Cuisine_type'],
        title='Top 5 Protein-Rich Recipes per Diet — Macro Distribution',
        labels={
            'Carbs(g)': 'Carbs (g)',
            'Protein(g)': 'Protein (g)',
            'Diet_type': 'Diet Type'
        },
        template='plotly_white',
        height=600
    )
    
    fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
    
    out_path = fig_dir / f"scatter_top5_{timestamp()}.html"
    fig.write_html(str(out_path), include_plotlyjs='cdn', div_id='scatter-plot')
    return out_path


def create_pie_chart(df: pd.DataFrame, fig_dir: Path, top_n: int = 8) -> Path:
    """Create interactive pie chart with Plotly"""
    counts = (
        df["Diet_type"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
        .value_counts()
        .sort_values(ascending=False)
    )
    
    if len(counts) > top_n:
        top = counts.iloc[:top_n]
        other_sum = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other_sum})])
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index.to_list(),
        values=counts.values,
        hole=0.3,
        hovertemplate='<b>%{label}</b><br>Recipes: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Recipe Distribution by Diet Type',
        template='plotly_white',
        height=500
    )
    
    out_path = fig_dir / f"pie_recipe_distribution_{timestamp()}.html"
    fig.write_html(str(out_path), include_plotlyjs='cdn', div_id='pie-chart')
    return out_path


def create_cluster_visualization(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Create interactive cluster visualization with Plotly"""
    features = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Protein vs Carbs', 'Protein vs Fat', 
                       'Carbs vs Fat', 'Elbow Method'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Plot 1: Protein vs Carbs
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['Protein(g)'],
                y=cluster_data['Carbs(g)'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=6, opacity=0.7),
                hovertemplate='<b>Cluster %{text}</b><br>Protein: %{x:.1f}g<br>Carbs: %{y:.1f}g<extra></extra>',
                text=[cluster] * len(cluster_data),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot 2: Protein vs Fat
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['Protein(g)'],
                y=cluster_data['Fat(g)'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=6, opacity=0.7),
                hovertemplate='<b>Cluster %{text}</b><br>Protein: %{x:.1f}g<br>Fat: %{y:.1f}g<extra></extra>',
                text=[cluster] * len(cluster_data),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Plot 3: Carbs vs Fat
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['Carbs(g)'],
                y=cluster_data['Fat(g)'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=6, opacity=0.7),
                hovertemplate='<b>Cluster %{text}</b><br>Carbs: %{x:.1f}g<br>Fat: %{y:.1f}g<extra></extra>',
                text=[cluster] * len(cluster_data),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Plot 4: Elbow Method
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(features_scaled)
        inertia.append(km.inertia_)
    
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=inertia,
            mode='lines+markers',
            marker=dict(size=8, color='blue'),
            line=dict(color='blue'),
            name='Inertia',
            showlegend=False,
            hovertemplate='k=%{x}<br>Inertia=%{y:.2f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add vertical line for optimal k
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", 
                  annotation_text=f"Optimal k={optimal_k}", row=2, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Protein (g)", row=1, col=1)
    fig.update_yaxes(title_text="Carbs (g)", row=1, col=1)
    fig.update_xaxes(title_text="Protein (g)", row=1, col=2)
    fig.update_yaxes(title_text="Fat (g)", row=1, col=2)
    fig.update_xaxes(title_text="Carbs (g)", row=2, col=1)
    fig.update_yaxes(title_text="Fat (g)", row=2, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=2, col=2)
    fig.update_yaxes(title_text="Inertia", row=2, col=2)
    
    fig.update_layout(
        title_text="Cluster Analysis",
        height=900,
        template='plotly_white',
        showlegend=True
    )
    
    out_path = fig_dir / f"clusters_{timestamp()}.html"
    fig.write_html(str(out_path), include_plotlyjs='cdn', div_id='cluster-chart')
    
    # Save cluster summary
    cluster_summary = df.groupby('Cluster').agg({
        'Protein(g)': ['mean', 'std'],
        'Carbs(g)': ['mean', 'std'],
        'Fat(g)': ['mean', 'std'],
        'Recipe_name': 'count'
    }).round(2)
    
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    cluster_summary = cluster_summary.rename(columns={'Recipe_name_count': 'Recipe_Count'})
    
    tables_dir = fig_dir.parent / "tables"
    tables_dir.mkdir(exist_ok=True)
    cluster_summary.to_csv(tables_dir / f"cluster_summary_{timestamp()}.csv")
    
    return out_path


def ensure_output_dirs(out_root: Path) -> Tuple[Path, Path]:
    tables = out_root / "tables"
    figs = out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    return tables, figs


def generate_cluster_figure():
    df = read_and_normalize(Path("All_Diets.csv"))
    df = add_engineered_features(df)

    features = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']]
    features_scaled = StandardScaler().fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)

    fig = px.scatter_3d(
        df,
        x="Protein(g)",
        y="Carbs(g)",
        z="Fat(g)",
        color=df["Cluster"].astype(str),
        hover_data=["Recipe_name", "Cuisine_type"],
        title="3D Cluster Visualization of Recipes",
    )
    fig.update_layout(template="plotly_white", height=600)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Task 1: Dataset analysis with interactive Plotly visualizations")
    parser.add_argument("csv_path", type=str, help="Path to All_Diets.csv")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory (default: outputs)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out)

    if not csv_path.exists():
        print(f"[{dt.datetime.now()}] ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[{dt.datetime.now()}] Loading CSV from: {csv_path}")
    df = read_and_normalize(csv_path)
    df = add_engineered_features(df)

    print(f"[{dt.datetime.now()}] Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    avg_macros, top5, highest, cuisines = compute_insights(df)

    tables_dir, figs_dir = ensure_output_dirs(out_dir)

    print(f"[{dt.datetime.now()}] Saving tables to: {tables_dir}")
    save_tables(out_dir, avg_macros, top5, highest, cuisines)

    print(f"[{dt.datetime.now()}] Generating interactive Plotly charts to: {figs_dir}")
    bar_path = create_bar_chart(avg_macros, figs_dir)
    heat_path = create_heatmap(avg_macros, figs_dir)
    scatter_path = create_scatter_plot(top5, figs_dir)
    pie_path = create_pie_chart(df, figs_dir)
    cluster_path = create_cluster_visualization(df, figs_dir)

    print("\n=== SUMMARY ===")
    print(f"Timestamp: {dt.datetime.now()}")
    print("\nSaved interactive charts:")
    print(" -", bar_path)
    print(" -", heat_path)
    print(" -", scatter_path)
    print(" -", pie_path)
    print(" -", cluster_path)
    print("================")


if __name__ == "__main__":
    main()
    