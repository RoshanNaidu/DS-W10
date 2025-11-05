# Exercise 1

URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv(URL)

# Select features and target variable
X = data[['100g_USD']]
y = data['rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('model_1.pickle', 'wb') as f:
    pickle.dump(model, f)



# Exercise 2

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Function to map roast category
def roast_category(roast):
    # Convert roast string values to numeric categories
    if pd.isnull(roast):
        return None
    roast_map = 
    {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    return roast_map.get(roast, None)

# Apply mapping to create roast_cat column
data['roast_cat'] = data['roast'].apply(roast_category)

# Select features and target for Decision Tree model
X = data[['100g_USD', 'roast_cat']]
y = data['rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Regressor
print("🔹 Training Decision Tree Regressor model...")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Save model
with open('model_2.pickle', 'wb') as f:
    pickle.dump(dt_model, f)
