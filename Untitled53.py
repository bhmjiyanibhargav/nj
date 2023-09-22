#!/usr/bin/env python
# coding: utf-8

# # question 01

# In[ ]:


# Step 1: Load and Preprocess the Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('heart_disease_dataset.csv')  # Replace with the actual file name

# Assuming 'target' column contains the labels (0 for no risk, 1 for risk)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators and other hyperparameters

# Train the classifier
clf.fit(X_train, y_train)

# Step 3: Evaluate the Model
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Step 4: Hyperparameter Tuning (Optional)
# You can use techniques like GridSearchCV or RandomizedSearchCV to perform hyperparameter tuning.
# Check for missing values

missing_values = data.isnull().sum()
print(missing_values)

# Assuming 'data' is your DataFrame
data.fillna(data.mean(), inplace=True)  # Replace NaN values with the mean of the column

data = pd.get_dummies(data, columns=['sex', 'chest_pain_type'], drop_first=True)




# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate']

data[numerical_features] = scaler.fit_transform(data[numerical_features])


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Preprocess the Dataset
data = pd.read_csv('heart_disease_dataset.csv')  # Replace with the actual file name

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Assuming 'target' column contains the labels (0 for no risk, 1 for risk)
X = data.drop('target', axis=1)
y = data['target']

# Fill missing values with the mean of the column
X.fillna(X.mean(), inplace=True)

# Encode categorical variables using one-hot encoding
X = pd.get_dummies(X, columns=['sex', 'chest_pain_type'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier with specified hyperparameters
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# In[ ]:


import matplotlib.pyplot as plt

# Get feature importances
feature_importances = clf.feature_importances_

# Create a DataFrame to associate feature names with their importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Get the top 5 most important features
top_features = importance_df.head(5)

# Print the top features
print(top_features)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 5 Most Important Features')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their values to search
param_grid = {
    'n_estimators': [50, 100, 150],            # Different number of trees
    'max_depth': [5, 10, 15],                 # Different maximum depths
    'min_samples_split': [2, 5, 10],         # Different minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]            # Different minimum samples required to be a leaf node
}

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Initialize Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# In[ ]:


# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
# Evaluate the default model on the test set
y_pred_default = clf.predict(X_test)

# Calculate evaluation metrics
accuracy_default = accuracy_score(y_test, y_pred_default)
precision_default = precision_score(y_test, y_pred_default)
recall_default = recall_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default)

# Print the evaluation metrics for the default model
print(f'Default Model Metrics:')
print(f'Accuracy: {accuracy_default}')
print(f'Precision: {precision_default}')
print(f'Recall: {recall_default}')
print(f'F1 Score: {f1_default}')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming 'top_features' contains the top 2 most important features
feature1 = top_features.iloc[0]['Feature']
feature2 = top_features.iloc[1]['Feature']

# Extract the corresponding columns from the test set
X_test_subset = X_test[[feature1, feature2]]

# Define a mesh grid for plotting
x_min, x_max = X_test_subset[feature1].min() - 1, X_test_subset[feature1].max() + 1
y_min, y_max = X_test_subset[feature2].min() - 1, X_test_subset[feature2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Get predictions for each point in the mesh grid
Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries and the scatter plot
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_test_subset[feature1], X_test_subset[feature2], c=y_test, edgecolors='k', marker='o', s=50, linewidth=1)

# Set labels and title
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('Decision Boundaries of Random Forest Classifier')

# Add a legend
plt.colorbar()

# Show the plot
plt.show()

