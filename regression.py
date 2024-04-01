import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("merged.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Select the relevant columns for regression
X = df[['beds']]
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model to the training data
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict the target values for the test data
y_pred = reg.predict(X_test)

# Calculate the R-squared value to evaluate the model's performance
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# Plot the predicted vs actual values to visualize the model's performance
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Regression Analysis Results")

# Add a line of best fit
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.show()