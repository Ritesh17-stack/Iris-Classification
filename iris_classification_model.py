# Importing Libraries
import pickle
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('IRIS.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(min(X[:, 3]), max(X[:, 3]))

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Training the Logistic regression model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Comparing the predicted and actual values
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Checking accuracy
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
# Checking for overfitting
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())

# Saving the model
filename = 'model.sav'
try:
    with open(filename, 'wb') as file:
        pickle.dump(classifier, file)
    print('Model Successfully saved')
except Exception as e:
    print(f'Error Saving the model : {str(e)}')

load_model = pickle.load(open(filename, 'rb'))
print(load_model.predict([[6.0, 2.1, 3.9, 1.1]]))
