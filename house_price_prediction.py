# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset
data = pd.read_csv("train.csv")

# Step 2: Select important columns
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Rename columns for simplicity
data.columns = ['square_feet', 'bedrooms', 'bathrooms', 'price']

# Step 3: Remove missing values
data = data.dropna()

# Step 4: Display dataset preview
print("Dataset Preview:")
print(data.head())

# Step 5: Define features and target
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']

# Step 6: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Create Linear Regression model
model = LinearRegression()

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Predict house prices
predictions = model.predict(X_test)

# Step 10: Evaluate model
mse = mean_squared_error(y_test, predictions)

print("\nModel Evaluation")
print("Mean Squared Error:", mse)

# Step 11: Predict price for a new house
new_house = pd.DataFrame([[2000,3,2]], columns=['square_feet','bedrooms','bathrooms'])
predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:", predicted_price)
