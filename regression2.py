import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('merged.csv')

# remove outliers
z_scores = stats.zscore(data['price'])
threshold = 3
data_no_outliers = data[(z_scores < threshold)]

# split the data into training and testing sets
X = data_no_outliers[['beds']] # independent variables
y = data_no_outliers['price'] # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the independent variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train the linear regression model on the scaled training data
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# returns the r-squared
score = model.score(X_test_scaled, y_test)
print('R^2 score:', score)

# predict the target variable using the trained model and the scaled independent variables
prediction = model.predict(X_test_scaled)
results = pd.DataFrame({'Predicted': prediction, 'Actual': y_test})

# create a scatter plot of the predicted vs. actual target variable values
plt.scatter(results['Actual'], results['Predicted'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()