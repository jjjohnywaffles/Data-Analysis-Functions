import pandas as pd
import statsmodels.api as sm

# Load the data into a Pandas DataFrame
df = pd.read_csv("merged.csv")

# Create dummy variables for the state column
dummies = pd.get_dummies(df["state"], prefix="state")

# Combine the dummy variables with the other columns
X = pd.concat([df[["beds", "price"]], dummies], axis=1)

# Define the target variable
y = df["price"]

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())