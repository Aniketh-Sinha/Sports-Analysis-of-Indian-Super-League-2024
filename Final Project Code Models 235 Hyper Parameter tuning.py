#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# Specify the file path
excel_file_path = r'C:\Users\SONY\OneDrive\Documents\New Data 10.xlsx'

# Print the absolute path
absolute_path = os.path.abspath(excel_file_path)
print(f"Absolute path: {absolute_path}")

# Check if the file exists
if os.path.exists(absolute_path):
    print("File exists.")
else:
    print("File does not exist.")
    
    
import pandas as pd

# Specify the path to your Excel file
excel_file_path = r'C:\Users\SONY\OneDrive\Documents\New Data 10.xlsx'

# Read the Excel file into a DataFrame
df5 = pd.read_excel(excel_file_path)

# Display the DataFrame
print(df5)


# In[2]:


df5.columns


# In[3]:


# Filter data for the year 2023
df_2023 = df5[df5['Year'] == 2023].copy()


# In[4]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np

# Load your DataFrame df containing the data for training and testing
# Make sure df contains features and target variable (future ratings of players)

# Filter data for training (2018 to 2022) and testing (2023)
train_data = df5[df5['Year'].between(2018, 2022)].copy()  # Ensure making a copy to avoid SettingWithCopyWarning
test_data = df5[df5['Year'] == 2023].copy()  # Ensure making a copy to avoid SettingWithCopyWarning

# Select relevant columns for training
features = ['MP_x', 'Starts', 'Min_x', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1',
            'Ast.1', 'G+A.1', 'G-PK.1', 'G+A-PK', 'GA90', 'SoTA', 'Saves', 'Save%', 'CS', 'CS%', 'PKA', 'PKsv',
            'PKm', 'Save%.1', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Mn/MP', 'Min%', 'Mn/Start',
            'Compl', 'Subs', 'Mn/Sub', 'unSub', 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', '2CrdY', 'Fls',
            'Fld', 'Off', 'Crs', 'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'MP_y', 'W', 'D', 'L', 'GF', 'GA', 'GD',
            'Pts', 'Pts/MP', 'Attendance']

target = 'PlayerRating'  # Change 'FutureRating' to the actual target column name

# Split the data into features (X) and target variable (y) for training and testing
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest model
model = RandomForestRegressor(random_state=42)

# Perform Grid Search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Predict player ratings for the test data using the best model
y_pred = best_model.predict(X_test)
test_data.loc[:, 'Predicted_Rating'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Hyperparameter Tuning:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating']])



# In[ ]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Load your DataFrame df containing the data for training and testing
# Make sure df contains features and target variable (future ratings of players)

# Filter data for training (2018 to 2022) and testing (2023)
train_data = df5[df5['Year'].between(2018, 2022)].copy()  # Ensure making a copy to avoid SettingWithCopyWarning
test_data = df5[df5['Year'] == 2023].copy()  # Ensure making a copy to avoid SettingWithCopyWarning

# Select relevant columns for training
features = ['MP_x', 'Starts', 'Min_x', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1',
            'Ast.1', 'G+A.1', 'G-PK.1', 'G+A-PK', 'GA90', 'SoTA', 'Saves', 'Save%', 'CS', 'CS%', 'PKA', 'PKsv',
            'PKm', 'Save%.1', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Mn/MP', 'Min%', 'Mn/Start',
            'Compl', 'Subs', 'Mn/Sub', 'unSub', 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', '2CrdY', 'Fls',
            'Fld', 'Off', 'Crs', 'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'MP_y', 'W', 'D', 'L', 'GF', 'GA', 'GD',
            'Pts', 'Pts/MP', 'Attendance']

target = 'PlayerRating'  # Change 'FutureRating' to the actual target column name

# Split the data into features (X) and target variable (y) for training and testing
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Define the hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)

# Perform Grid Search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Predict player ratings for the test data using the best model
y_pred = best_model.predict(X_test)
test_data.loc[:, 'Predicted_Rating'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Hyperparameter Tuning for Gradient Boosting:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating']])


# In[ ]:


import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Load your DataFrame df containing the data for training and testing
# Make sure df contains features and target variable (future ratings of players)

# Filter data for training (2018 to 2022) and testing (2023)
train_data = df5[df5['Year'].between(2018, 2022)].copy()  # Ensure making a copy to avoid SettingWithCopyWarning
test_data = df5[df5['Year'] == 2023].copy()  # Ensure making a copy to avoid SettingWithCopyWarning

# Select relevant columns for training
features = ['MP_x', 'Starts', 'Min_x', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1',
            'Ast.1', 'G+A.1', 'G-PK.1', 'G+A-PK', 'GA90', 'SoTA', 'Saves', 'Save%', 'CS', 'CS%', 'PKA', 'PKsv',
            'PKm', 'Save%.1', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Mn/MP', 'Min%', 'Mn/Start',
            'Compl', 'Subs', 'Mn/Sub', 'unSub', 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', '2CrdY', 'Fls',
            'Fld', 'Off', 'Crs', 'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'MP_y', 'W', 'D', 'L', 'GF', 'GA', 'GD',
            'Pts', 'Pts/MP', 'Attendance']

target = 'PlayerRating'  # Change 'FutureRating' to the actual target column name

# Split the data into features (X) and target variable (y) for training and testing
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Define the hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 1]
}

# Initialize the XGBoost model
model = xgb.XGBRegressor(random_state=42)

# Perform Grid Search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Predict player ratings for the test data using the best model
y_pred = best_model.predict(X_test)
test_data.loc[:, 'Predicted_Rating'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Hyperparameter Tuning for XGBoost:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating']])


# In[ ]:


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Load your DataFrame df containing the data for training and testing
# Make sure df contains features and target variable (future ratings of players)

# Filter data for training (2018 to 2022) and testing (2023)
train_data = df5[df5['Year'].between(2018, 2022)].copy()  # Ensure making a copy to avoid SettingWithCopyWarning
test_data = df5[df5['Year'] == 2023].copy()  # Ensure making a copy to avoid SettingWithCopyWarning

# Select relevant columns for training
features = ['MP_x', 'Starts', 'Min_x', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1',
            'Ast.1', 'G+A.1', 'G-PK.1', 'G+A-PK', 'GA90', 'SoTA', 'Saves', 'Save%', 'CS', 'CS%', 'PKA', 'PKsv',
            'PKm', 'Save%.1', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Mn/MP', 'Min%', 'Mn/Start',
            'Compl', 'Subs', 'Mn/Sub', 'unSub', 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', '2CrdY', 'Fls',
            'Fld', 'Off', 'Crs', 'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'MP_y', 'W', 'D', 'L', 'GF', 'GA', 'GD',
            'Pts', 'Pts/MP', 'Attendance']

target = 'PlayerRating'  # Change 'FutureRating' to the actual target column name

# Split the data into features (X) and target variable (y) for training and testing
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Define the hyperparameters grid
param_grid = {
    'alpha': [0.1, 1, 10, 100]
}

# Initialize the Ridge Regression model
model = Ridge()

# Perform Grid Search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Predict player ratings for the test data using the best model
y_pred = best_model.predict(X_test)
test_data.loc[:, 'Predicted_Rating'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Hyperparameter Tuning for Ridge Regression:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating']])


# In[ ]:


import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Load your DataFrame df containing the data for training and testing
# Make sure df contains features and target variable (future ratings of players)

# Filter data for training (2018 to 2022) and testing (2023)
train_data = df5[df5['Year'].between(2018, 2022)].copy()  # Ensure making a copy to avoid SettingWithCopyWarning
test_data = df5[df5['Year'] == 2023].copy()  # Ensure making a copy to avoid SettingWithCopyWarning

# Select relevant columns for training
features = ['MP_x', 'Starts', 'Min_x', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1',
            'Ast.1', 'G+A.1', 'G-PK.1', 'G+A-PK', 'GA90', 'SoTA', 'Saves', 'Save%', 'CS', 'CS%', 'PKA', 'PKsv',
            'PKm', 'Save%.1', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Mn/MP', 'Min%', 'Mn/Start',
            'Compl', 'Subs', 'Mn/Sub', 'unSub', 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', '2CrdY', 'Fls',
            'Fld', 'Off', 'Crs', 'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'MP_y', 'W', 'D', 'L', 'GF', 'GA', 'GD',
            'Pts', 'Pts/MP', 'Attendance']

target = 'PlayerRating'  # Change 'FutureRating' to the actual target column name

# Split the data into features (X) and target variable (y) for training and testing
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Define the hyperparameters grid
param_grid = {
    'alpha': [0.1, 1, 10, 100]
}

# Initialize the Lasso Regression model
model = Lasso()

# Perform Grid Search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Predict player ratings for the test data using the best model
y_pred = best_model.predict(X_test)
test_data.loc[:, 'Predicted_Rating'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Hyperparameter Tuning for Lasso Regression:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating']])


# In[ ]:




