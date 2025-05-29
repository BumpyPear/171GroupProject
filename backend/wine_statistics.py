import numpy as np
from ucimlrepo import fetch_ucirepo

# Fetch the wine quality dataset
wine_quality = fetch_ucirepo(id=186)
df_full = wine_quality.data.original
df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)
X = df_white.drop(columns=['quality', 'color'])
y = df_white['quality']

# Extract data into lists
fixed_acidity = X['fixed_acidity'].tolist()
volatile_acidity = X['volatile_acidity'].tolist()
citric_acid = X['citric_acid'].tolist()
residual_sugar = X['residual_sugar'].tolist()
chlorides = X['chlorides'].tolist()
free_sulfur_dioxide = X['free_sulfur_dioxide'].tolist()
total_sulfur_dioxide = X['total_sulfur_dioxide'].tolist()
density = X['density'].tolist()
pH = X['pH'].tolist()
sulphates = X['sulphates'].tolist()
alcohol = X['alcohol'].tolist()
quality = y.tolist()

List_of_lists = []
List_of_lists.append(fixed_acidity)
List_of_lists.append(volatile_acidity)
List_of_lists.append(citric_acid)
List_of_lists.append(residual_sugar)
List_of_lists.append(chlorides)
List_of_lists.append(free_sulfur_dioxide)
List_of_lists.append(total_sulfur_dioxide)
List_of_lists.append(density)
List_of_lists.append(pH)
List_of_lists.append(sulphates)
List_of_lists.append(alcohol)
List_of_lists.append(quality)

List_of_names = []
List_of_names.append("Fixed Acidity")
List_of_names.append("Volatile Acidity")
List_of_names.append("Citric Acid")
List_of_names.append("Residual Sugar")
List_of_names.append("Chlorides")
List_of_names.append("Free Sulfur Dioxide")
List_of_names.append("Total Sulfur Dioxide")
List_of_names.append("Density")
List_of_names.append("pH")
List_of_names.append("Sulphates")
List_of_names.append("Alcohol")
List_of_names.append("Quality")

counter = 0
for sublist in List_of_lists:
    print(f"Statistics for {List_of_names[counter]}:")
    print(f"Median: {np.median(sublist)}")
    print(f"Mean: {np.mean(sublist)}")
    print(f"Standard deviation: {np.std(sublist)}")
    print(f"Variance: {np.var(sublist)}")
    print(f"Minimum: {np.min(sublist)}")
    print(f"Maximum: {np.max(sublist)}")
    counter += 1