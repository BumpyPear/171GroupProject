import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)

df_full = wine_quality.data.original

df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)

X = df_white.drop(columns=['quality', 'color'])
y = df_white['quality']

# IQR method for outlier detection
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
lower_bounds = Q1 - 1.5 * IQR
upper_bounds = Q3 + 1.5 * IQR

# Create box plots to visualize outliers
plt.figure(figsize=(16, 12))
plt.suptitle('Box Plots for Outliers in White Wine Features', fontsize=16)

# Plot each feature
for i, feature in enumerate(X.columns, 1):
    plt.subplot(4, 3, i)
    plt.boxplot(X[feature], vert=True, patch_artist=True, 
               showfliers=True, boxprops=dict(facecolor='skyblue'))
    plt.title(feature)
    plt.xlabel(feature)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('white_wine_boxplots_outliers.png')

# Find outliers: values outside Q1-1.5*IQR or Q3+1.5*IQR
outliers = (X < lower_bounds) | (X > upper_bounds)

# Print statistics and sort features by outlier count (descending)
print(f"{'Feature':<20} {'Outlier Count':<18} {'Percentage':<12} {'Bounds'}\n")
outlier_counts = outliers.sum().sort_values(ascending=False)

for feature in outlier_counts.index:
    count = outliers[feature].sum()
    percentage = (count / len(X)) * 100
    lower = lower_bounds[feature]
    upper = upper_bounds[feature]
    print("{:<20} {:<10} {:14.2f}%  [{:.2f}, {:.2f}]".format(feature, count, percentage, lower, upper))