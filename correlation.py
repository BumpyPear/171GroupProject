import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets

df = X.copy()
df["quality"] = y.squeeze()          

# Feature vs target correlation
target_corr = (
    df.corr(numeric_only=True)["quality"]
      .drop("quality")
      .sort_values(ascending=False)
)
print("Feature vs quality (Pearson):\n", target_corr, "\n")

# Feature vs feature correlation
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# columns that have any correlation above the cutoff (say, 0.80)
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
plt.title("Wine‑quality correlation matrix (Pearson)")
plt.show()
