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


# In[ ]:


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
from sklearn.feature_selection import RFECV

# Define the Random Forest model
model = RandomForestRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv = RFECV(estimator=model, cv=5, scoring='neg_mean_squared_error')
rfecv.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices = rfecv.support_

# Filter the training and test data with selected features
X_train_selected = X_train.iloc[:, selected_features_indices]
X_test_selected = X_test.iloc[:, selected_features_indices]

# Perform Grid Search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_selected, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train_selected, y_train)

# Predict player ratings for the test data
y_pred = best_model.predict(X_test_selected)
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
print("Evaluation Results after Feature Selection (RFECV) and Hyperparameter Tuning:")
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
from sklearn.feature_selection import RFECV

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

# Define the Gradient Boosting model
model_gb = GradientBoostingRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_gb = RFECV(estimator=model_gb, cv=5, scoring='neg_mean_squared_error')
rfecv_gb.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_gb = rfecv_gb.support_

# Filter the training and test data with selected features
X_train_selected_gb = X_train.iloc[:, selected_features_indices_gb]
X_test_selected_gb = X_test.iloc[:, selected_features_indices_gb]

# Perform Grid Search to find the best combination of hyperparameters
grid_search_gb = GridSearchCV(estimator=model_gb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train_selected_gb, y_train)

# Get the best model from grid search
best_model_gb = grid_search_gb.best_estimator_

# Train the best model
best_model_gb.fit(X_train_selected_gb, y_train)

# Predict player ratings for the test data
y_pred_gb = best_model_gb.predict(X_test_selected_gb)
test_data.loc[:, 'Predicted_Rating_gb'] = y_pred_gb  # Use .loc to set values

# Calculate evaluation metrics
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating_gb', ascending=False)
test_data['Predicted_Rank_gb'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Feature Selection (RFECV) and Hyperparameter Tuning for Gradient Boosting:")
print(f"  MAE: {mae_gb:.4f}")
print(f"  MSE: {mse_gb:.4f}")
print(f"  RMSE: {rmse_gb:.4f}")
print(f"  R2: {r2_gb:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_gb']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_gb']])



# In[ ]:


import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

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

# Define the XGBoost model
model_xgb = XGBRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_xgb = RFECV(estimator=model_xgb, cv=5, scoring='neg_mean_squared_error')
rfecv_xgb.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_xgb = rfecv_xgb.support_

# Filter the training and test data with selected features
X_train_selected_xgb = X_train.iloc[:, selected_features_indices_xgb]
X_test_selected_xgb = X_test.iloc[:, selected_features_indices_xgb]

# Perform Grid Search to find the best combination of hyperparameters
grid_search_xgb = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train_selected_xgb, y_train)

# Get the best model from grid search
best_model_xgb = grid_search_xgb.best_estimator_

# Train the best model
best_model_xgb.fit(X_train_selected_xgb, y_train)

# Predict player ratings for the test data
y_pred_xgb = best_model_xgb.predict(X_test_selected_xgb)
test_data.loc[:, 'Predicted_Rating_xgb'] = y_pred_xgb  # Use .loc to set values

# Calculate evaluation metrics
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating_xgb', ascending=False)
test_data['Predicted_Rank_xgb'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Feature Selection (RFECV) and Hyperparameter Tuning for XGBoost:")
print(f"  MAE: {mae_xgb:.4f}")
print(f"  MSE: {mse_xgb:.4f}")
print(f"  RMSE: {rmse_xgb:.4f}")
print(f"  R2: {r2_xgb:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_xgb']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_xgb']])


# In[ ]:


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Ridge Regression model
model_ridge = Ridge()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.1, 1, 10],
}

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_ridge = RFECV(estimator=model_ridge, cv=5, scoring='neg_mean_squared_error')
rfecv_ridge.fit(X_train_scaled, y_train)

# Get selected feature indices
selected_features_indices_ridge = rfecv_ridge.support_

# Filter the training and test data with selected features
X_train_selected_ridge = X_train_scaled[:, selected_features_indices_ridge]
X_test_selected_ridge = X_test_scaled[:, selected_features_indices_ridge]

# Perform Grid Search to find the best combination of hyperparameters
grid_search_ridge = GridSearchCV(estimator=model_ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_ridge.fit(X_train_selected_ridge, y_train)

# Get the best model from grid search
best_model_ridge = grid_search_ridge.best_estimator_

# Train the best model
best_model_ridge.fit(X_train_selected_ridge, y_train)

# Predict player ratings for the test data
y_pred_ridge = best_model_ridge.predict(X_test_selected_ridge)
test_data.loc[:, 'Predicted_Rating_ridge'] = y_pred_ridge  # Use .loc to set values

# Calculate evaluation metrics
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating_ridge', ascending=False)
test_data['Predicted_Rank_ridge'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Feature Selection (RFECV) and Hyperparameter Tuning for Ridge Regression:")
print(f"  MAE: {mae_ridge:.4f}")
print(f"  MSE: {mse_ridge:.4f}")
print(f"  RMSE: {rmse_ridge:.4f}")
print(f"  R2: {r2_ridge:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_ridge']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_ridge']])


# In[ ]:


import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Lasso Regression model
model_lasso = Lasso()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.1, 1, 10],
}

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_lasso = RFECV(estimator=model_lasso, cv=5, scoring='neg_mean_squared_error')
rfecv_lasso.fit(X_train_scaled, y_train)

# Get selected feature indices
selected_features_indices_lasso = rfecv_lasso.support_

# Filter the training and test data with selected features
X_train_selected_lasso = X_train_scaled[:, selected_features_indices_lasso]
X_test_selected_lasso = X_test_scaled[:, selected_features_indices_lasso]

# Perform Grid Search to find the best combination of hyperparameters
grid_search_lasso = GridSearchCV(estimator=model_lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train_selected_lasso, y_train)

# Get the best model from grid search
best_model_lasso = grid_search_lasso.best_estimator_

# Train the best model
best_model_lasso.fit(X_train_selected_lasso, y_train)

# Predict player ratings for the test data
y_pred_lasso = best_model_lasso.predict(X_test_selected_lasso)
test_data.loc[:, 'Predicted_Rating_lasso'] = y_pred_lasso  # Use .loc to set values

# Calculate evaluation metrics
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating_lasso', ascending=False)
test_data['Predicted_Rank_lasso'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Feature Selection (RFECV) and Hyperparameter Tuning for Lasso Regression:")
print(f"  MAE: {mae_lasso:.4f}")
print(f"  MSE: {mse_lasso:.4f}")
print(f"  RMSE: {rmse_lasso:.4f}")
print(f"  R2: {r2_lasso:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_lasso']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_lasso']])



# In[ ]:




