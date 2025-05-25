from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


wine_quality = fetch_ucirepo(id=186)

df_full = wine_quality.data.original

df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)

X = df_white.drop(columns=['quality', 'color'])
y = df_white['quality']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Original ranges (white wine only):")
print(X.describe().loc[['min', 'max']])
print("\nScaled ranges:")
print(X_scaled.describe().loc[['min', 'max']])

print("Missing values in features:\n", X.isnull().sum())
print("\nMissing values in target:\n", y.isnull().sum())
