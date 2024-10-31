from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

url = "https://raw.githubusercontent.com/businessdatasolutions/courses/main/data%20mining/gitbook/datasets/breastcancer.csv"
df = pd.read_csv(url)

#Data
X = df.drop(columns = ['diagnosis'])
y = df['diagnosis']

#Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Random forest 
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=11)

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

from sklearn.model_selection import GridSearchCV

# Create a RandomForestClassifier
clf = RandomForestClassifier(random_state=42)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [90,100,110,120],
    'max_depth': [1,2,3,4,5,6]
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search
grid_search.fit(X_train, y_train)

# Output the best parameters and accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Test on the test set with the best model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
print(f'Test Set Accuracy: {accuracy_score(y_test, y_pred)*100:.4f}%')
