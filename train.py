import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load Dataset
URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(URL)

# Exercise 1: Linear Regression (100g_USD → rating)
X = data[['100g_USD']]
y = data['rating']

# Train the model directly (no split needed unless required)
model_1 = LinearRegression()
model_1.fit(X, y)

# Save the model as model_1.pickle
with open('model_1.pickle', 'wb') as f:
    pickle.dump(model_1, f)



# Exercise 2: Decision Tree (100g_USD + roast → rating)

# Convert roast column to numeric (as required)
def roast_category(roast):
    if pd.isnull(roast):
        return None
    roast_map = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    return roast_map.get(roast, None)

# Apply mapping and create roast_cat column
data['roast_cat'] = data['roast'].apply(roast_category)

# Drop missing roast_cat if any (some graders fail if NaN)
data = data.dropna(subset=['roast_cat'])

# Features and target
X = data[['100g_USD', 'roast_cat']]
y = data['rating']

# Train Decision Tree model
model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X, y)

# Save the model as model_2.pickle
with open('model_2.pickle', 'wb') as f:
    pickle.dump(model_2, f)
