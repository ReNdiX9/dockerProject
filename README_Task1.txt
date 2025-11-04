
# Task 1 Runner (All_Diets.csv)

**How to use (in your Azure Ubuntu VM):**
1. Put `All_Diets.csv` somewhere on disk (e.g., `~/data/All_Diets.csv`).
2. Install deps if needed:
   ```bash
   python3 -m pip install pandas matplotlib numpy
   ```
3. Run:
   ```bash
   python3 data_analysis.py ~/data/All_Diets.csv --out outputs
   ```

**What you get:**
- `outputs/tables/*.csv` with:
  - `avg_macros_by_diet_*.csv`
  - `top5_protein_by_diet_*.csv`
  - `diet_highest_mean_protein_*.csv`
  - `cuisine_counts_by_diet_*.csv`
- `outputs/figures/*.png` with 3 charts (bar, heatmap, scatter).
- Console summary includes a timestamp—perfect for screenshots with date/time visible.

**Notes:**
- Script uses Matplotlib only (assignment-friendly).
- Handles missing numeric values by filling with the mean.
- Safely computes engineered ratios without divide-by-zero.


# app.py packages
pip install flask pandas numpy matplotlib scikit-learn
# data_analysis packages
pip install pandas numpy matplotlib
# Run to generate charts - python data_analysis.py ./All_Diets.csv --out ./outputs
# run flask app python app.py