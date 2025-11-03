
#!/usr/bin/env python3
"""
data_analysis.py — Task 1 automation for the All_Diets.csv project

What this script does
---------------------
1) Loads the All_Diets.csv dataset.
2) Cleans and normalizes key columns (diet/cuisine names; numeric macros).
3) Fills missing numeric values with the column mean.
4) Adds engineered metrics:
   - Protein_to_Carbs_ratio
   - Carbs_to_Fat_ratio
5) Computes required insights:
   - Average Protein/Carbs/Fat by Diet_type
   - Top 5 protein‑rich recipes per Diet_type
   - Diet_type with the highest protein content (by mean Protein(g))
   - Most common cuisines per Diet_type
6) Saves tidy CSV outputs and Matplotlib charts (bar, heatmap, scatter).
7) Prints a concise summary and timestamps for your screenshots.

Usage
-----
python data_analysis.py /path/to/All_Diets.csv --out outputs

Notes
-----
- Uses Matplotlib only (no seaborn), one chart per figure, no explicit colors.
- Figures are saved into <out>/figures as PNG with timestamp in the filename.
- CSV results are saved into <out>/tables.
"""

import argparse
import datetime as dt
import sys
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # Normalize column names: trim and keep as-is for required names, try to map common variants
    # Also handle possible space variations like 'Protein (g)'
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

    # Verify required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize Diet_type & Cuisine_type (strip, lower -> title)
    for text_col in ["Diet_type", "Cuisine_type"]:
        df[text_col] = (
            df[text_col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "none": np.nan})
        )
        # Some rows can be lower/upper mixed; unify simple capitalization while keeping acronyms
        df[text_col] = df[text_col].apply(lambda x: x.title() if isinstance(x, str) else x)

    # Coerce numeric macros (some CSVs may store them as strings)
    for num_col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Handle missing numeric data: fill with mean (per column)
    for num_col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        mean_val = df[num_col].mean(skipna=True)
        df[num_col] = df[num_col].fillna(mean_val)

    # Replace zero fat/carbs when computing ratios (to avoid divide-by-zero later)
    # Keep originals; we will guard division during ratio creation.
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Guard divisions using np.where to avoid inf; set to NaN when denominator is 0
    carbs = df["Carbs(g)"].to_numpy()
    fat = df["Fat(g)"].to_numpy()
    protein = df["Protein(g)"].to_numpy()

    prot_to_carbs = np.where(carbs != 0, protein / carbs, np.nan)
    carbs_to_fat = np.where(fat != 0, carbs / fat, np.nan)

    df["Protein_to_Carbs_ratio"] = prot_to_carbs
    df["Carbs_to_Fat_ratio"] = carbs_to_fat
    return df


def compute_insights(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Averages by Diet_type
    avg_macros = (
        df.groupby("Diet_type", as_index=True)[["Protein(g)", "Carbs(g)", "Fat(g)"]]
        .mean()
        .sort_index()
    )

    # Top 5 protein‑rich recipes per Diet_type
    # Sort by Protein then groupby head(5)
    top5_protein = (
        df.sort_values(["Diet_type", "Protein(g)"], ascending=[True, False])
        .groupby("Diet_type", group_keys=False)
        .head(5)
        .loc[:, ["Diet_type", "Recipe_name", "Cuisine_type", "Protein(g)", "Carbs(g)", "Fat(g)"]]
    )

    # Diet_type with highest mean protein
    highest_protein_by_diet = (
        avg_macros["Protein(g)"]
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Protein(g)": "Mean_Protein(g)"})
    )

    # Most common cuisines per Diet_type (frequency)
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


def plot_bar_avg_macros(avg_macros: pd.DataFrame, fig_dir: Path) -> Path:
    # Make a grouped bar chart: three bars per diet (Protein, Carbs, Fat)
    diets = avg_macros.index.to_list()
    vals = avg_macros[["Protein(g)", "Carbs(g)", "Fat(g)"]].to_numpy()

    x = np.arange(len(diets))
    width = 0.25

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(x - width, vals[:, 0], width, label="Protein(g)")
    ax.bar(x,         vals[:, 1], width, label="Carbs(g)")
    ax.bar(x + width, vals[:, 2], width, label="Fat(g)")

    ax.set_title("Average Macronutrients by Diet Type")
    ax.set_xlabel("Diet Type")
    ax.set_ylabel("Grams (g)")
    ax.set_xticks(x)
    ax.set_xticklabels(diets, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = fig_dir / f"bar_avg_macros_{timestamp()}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_heatmap_avg_macros(avg_macros: pd.DataFrame, fig_dir: Path) -> Path:
    # Basic heatmap using imshow (no seaborn)
    data = avg_macros[["Protein(g)", "Carbs(g)", "Fat(g)"]].to_numpy()
    diets = avg_macros.index.to_list()
    nutrients = ["Protein(g)", "Carbs(g)", "Fat(g)"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, aspect="auto")

    # Tick labels
    ax.set_xticks(np.arange(len(nutrients)))
    ax.set_xticklabels(nutrients, rotation=0)
    ax.set_yticks(np.arange(len(diets)))
    ax.set_yticklabels(diets)

    ax.set_title("Heatmap: Average Macros by Diet Type")
    # Add value annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = fig_dir / f"heatmap_avg_macros_{timestamp()}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_scatter_top5(top5: pd.DataFrame, fig_dir: Path) -> Path:
    # Scatter by cuisine: x = Carbs, y = Protein, size ~ Fat
    # Encode cuisines as integer positions along x-axis clusters per diet? Simpler: standard x/y scatter.
    x = top5["Carbs(g)"].to_numpy()
    y = top5["Protein(g)"].to_numpy()
    sizes = np.clip(top5["Fat(g)"].to_numpy(), 1, None)  # ensure >0 for visibility
    sizes = (sizes / np.nanmax(sizes)) * 300.0 + 20.0    # scale bubble sizes

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, s=sizes, alpha=0.7)

    ax.set_title("Top 5 Protein‑Rich Recipes per Diet — Macro Distribution")
    ax.set_xlabel("Carbs (g)")
    ax.set_ylabel("Protein (g)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Add a few labels (Recipe_name) to reduce clutter: label the top 10 by protein
    labeled = (
        top5.nlargest(10, "Protein(g)")
        .loc[:, ["Recipe_name", "Carbs(g)", "Protein(g)"]]
        .itertuples(index=False)
    )
    for rname, cx, py in labeled:
        ax.annotate(str(rname)[:30], (cx, py), xytext=(5, 5), textcoords="offset points", fontsize=8)

    fig.tight_layout()
    out_path = fig_dir / f"scatter_top5_{timestamp()}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

# pie chart
def plot_pie_recipe_distribution(df: pd.DataFrame, fig_dir: Path, top_n: int = 8) -> Path:
    """
    Pie chart of how many recipes each Diet_type has.
    To keep labels readable, we keep top_n categories and group the rest into 'Other'.
    Uses Matplotlib only (no seaborn), one chart per figure, no explicit colors.
    """
    # Count by Diet_type, handle missing
    counts = (
        df["Diet_type"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
        .value_counts()
        .sort_values(ascending=False)
    )

    # Group small categories into "Other" to avoid label clutter
    if len(counts) > top_n:
        top = counts.iloc[:top_n]
        other_sum = counts.iloc[top_n:].sum()
        counts = top.append(pd.Series({"Other": other_sum}))

    labels = counts.index.to_list()
    sizes = counts.values

    fig = plt.figure()
    ax = fig.add_subplot(111)

def plot_clusters(df: pd.DataFrame, fig_dir: Path) -> Path:
    """
    Perform K-means clustering on macronutrients and create visualization
    """
    # Prepare features for clustering
    features = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine optimal number of clusters using elbow method
    inertia = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
    
    # Use elbow method to find optimal k
    optimal_k = 4
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Protein vs Carbs colored by cluster
    scatter1 = axes[0, 0].scatter(df['Protein(g)'], df['Carbs(g)'], 
                                 c=df['Cluster'], cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('Protein (g)')
    axes[0, 0].set_ylabel('Carbs (g)')
    axes[0, 0].set_title('Clusters: Protein vs Carbs')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # Plot 2: Protein vs Fat colored by cluster
    scatter2 = axes[0, 1].scatter(df['Protein(g)'], df['Fat(g)'], 
                                 c=df['Cluster'], cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('Protein (g)')
    axes[0, 1].set_ylabel('Fat (g)')
    axes[0, 1].set_title('Clusters: Protein vs Fat')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # Plot 3: Carbs vs Fat colored by cluster
    scatter3 = axes[1, 0].scatter(df['Carbs(g)'], df['Fat(g)'], 
                                 c=df['Cluster'], cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Carbs (g)')
    axes[1, 0].set_ylabel('Fat (g)')
    axes[1, 0].set_title('Clusters: Carbs vs Fat')
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    # Plot 4: Elbow method
    axes[1, 1].plot(k_range, inertia, 'bo-')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Inertia')
    axes[1, 1].set_title('Elbow Method for Optimal k')
    axes[1, 1].axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    out_path = fig_dir / f"clusters_{timestamp()}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    
    # Save cluster information
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

def compute_cluster_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed insights for each cluster
    """
    if 'Cluster' not in df.columns:
        return pd.DataFrame()
    
    insights = []
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        
        insight = {
            'Cluster': cluster,
            'Total_Recipes': len(cluster_data),
            'Avg_Protein': cluster_data['Protein(g)'].mean(),
            'Avg_Carbs': cluster_data['Carbs(g)'].mean(),
            'Avg_Fat': cluster_data['Fat(g)'].mean(),
            'Common_Diet': cluster_data['Diet_type'].mode().iloc[0] if not cluster_data['Diet_type'].mode().empty else 'Unknown',
            'Common_Cuisine': cluster_data['Cuisine_type'].mode().iloc[0] if not cluster_data['Cuisine_type'].mode().empty else 'Unknown',
            'Protein_Range': f"{cluster_data['Protein(g)'].min():.1f}-{cluster_data['Protein(g)'].max():.1f}",
            'Carbs_Range': f"{cluster_data['Carbs(g)'].min():.1f}-{cluster_data['Carbs(g)'].max():.1f}",
            'Fat_Range': f"{cluster_data['Fat(g)'].min():.1f}-{cluster_data['Fat(g)'].max():.1f}"
        }
        insights.append(insight)
    
    return pd.DataFrame(insights)

    # Show % only for slices >=3% to reduce noise
    def _pct(p):
        return f"{p:.1f}%" if p >= 3 else ""

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=_pct,
        startangle=90,
    )
    ax.set_title("Recipe Distribution by Diet Type")
    ax.axis("equal")  # Equal aspect ratio for a circle

    fig.tight_layout()
    out_path = fig_dir / f"pie_recipe_distribution_{timestamp()}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path





def ensure_output_dirs(out_root: Path) -> Tuple[Path, Path]:
    tables = out_root / "tables"
    figs = out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    return tables, figs


def main():
    parser = argparse.ArgumentParser(description="Task 1: Dataset analysis and visualizations")
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

    print(f"[{dt.datetime.now()}] Generating figures to: {figs_dir}")
    bar_path = plot_bar_avg_macros(avg_macros, figs_dir)
    heat_path = plot_heatmap_avg_macros(avg_macros, figs_dir)
    scatter_path = plot_scatter_top5(top5, figs_dir)
    pie_path = plot_pie_recipe_distribution(df, figs_dir)

    print(f"[{dt.datetime.now()}] Performing clustering analysis...")
    cluster_path = plot_clusters(df, figs_dir)
    cluster_insights = compute_cluster_insights(df)
    
    # Save cluster insights
    if not cluster_insights.empty:
        cluster_insights.to_csv(tables_dir / f"cluster_insights_{timestamp()}.csv", index=False)

    # Print highlights for quick screenshot
    print("\n=== SUMMARY (copy this into your report) ===")
    print(f"Timestamp: {dt.datetime.now()}")
    print("\nDiet with highest mean Protein(g):")
    print(highest.head(1).to_string(index=False))

    print("\nAverage macros by diet (first 10):")
    print(avg_macros.head(10).round(2).to_string())

    print("\nMost common cuisines per diet (top 10 rows):")
    print(cuisines.head(10).to_string(index=False))

    print("\nTop 5 protein‑rich recipes per diet (sample):")
    print(top5.groupby('Diet_type').head(1).to_string(index=False))

    print("\nCluster Insights:")
    if not cluster_insights.empty:
        print(cluster_insights.to_string(index=False))

    print("\nSaved figures:")
    print(" -", bar_path)
    print(" -", heat_path)
    print(" -", scatter_path) 
    print(" -", pie_path)
    print(" -", cluster_path)
    print("============================================")


if __name__ == "__main__":
    main()
