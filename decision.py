import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the data into a Pandas DataFrame
data = pd.read_csv('merged.csv')

# split the data into training and testing sets
X = data[['beds']] # independent variables
y = data['price'] # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a decision tree classifier
model = DecisionTreeClassifier()

# train the decision tree classifier on the training data
model.fit(X_train, y_train)

# make predictions on the testing data
predictions = model.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)