import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Load dataset (includes both red and white)
wine_quality = fetch_ucirepo(id=186)

# Use the full version that includes 'color'
df_full = wine_quality.data.original

# Filter for white or red wine
df_white = df_full[df_full['color'] == 'red'].reset_index(drop=True)

# Separate features and target
X = df_white.drop(columns=['quality', 'color'])
y = df_white['quality']

# Create combined DataFrame for correlation analysis
df_corr = X.copy()
df_corr["quality"] = y

# Feature vs target correlation
target_corr = (
    df_corr.corr(numeric_only=True)["quality"]
      .drop("quality")
      .sort_values(ascending=False)
)
print("[RED WINE] Feature vs quality (Pearson):\n", target_corr, "\n")

# Feature vs feature correlation
corr_matrix = df_corr.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_cutoff = 0.80
to_consider_dropping = [
    col for col in upper.columns if any(upper[col] > high_corr_cutoff)
]
print("Potentially redundant features (>|0.80| with another column):",
      to_consider_dropping)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            vmin=-1, vmax=1,
            cmap="coolwarm",
            linewidths=.5, square=True)
plt.title("White Wine Quality Correlation Matrix (Pearson)")
plt.show()
