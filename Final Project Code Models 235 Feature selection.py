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

# Initialize the Random Forest model with fixed hyperparameters
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # Set hyperparameters as desired

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_rf = RFECV(estimator=model_rf, cv=5, scoring='neg_mean_squared_error')
rfecv_rf.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_rf = rfecv_rf.support_

# Filter the training and test data with selected features
X_train_selected_rf = X_train.iloc[:, selected_features_indices_rf]
X_test_selected_rf = X_test.iloc[:, selected_features_indices_rf]

# Train the Random Forest model with selected features
model_rf.fit(X_train_selected_rf, y_train)

# Predict player ratings for the test data
y_pred_rf = model_rf.predict(X_test_selected_rf)
test_data.loc[:, 'Predicted_Rating_rf'] = y_pred_rf  # Use .loc to set values

# Calculate evaluation metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating_rf', ascending=False)
test_data['Predicted_Rank_rf'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results after Feature Selection (RFECV) for Random Forest:")
print(f"  MAE: {mae_rf:.4f}")
print(f"  MSE: {mse_rf:.4f}")
print(f"  RMSE: {rmse_rf:.4f}")
print(f"  R2: {r2_rf:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_rf']]) 
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_rf']])




# In[7]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
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

# Initialize the Gradient Boosting model with fixed hyperparameters
model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)  # Set hyperparameters as desired

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_gb = RFECV(estimator=model_gb, cv=5, scoring='neg_mean_squared_error')
rfecv_gb.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_gb = rfecv_gb.support_

# Filter the training and test data with selected features
X_train_selected_gb = X_train.iloc[:, selected_features_indices_gb]
X_test_selected_gb = X_test.iloc[:, selected_features_indices_gb]

# Train the Gradient Boosting model with selected features
model_gb.fit(X_train_selected_gb, y_train)

# Predict player ratings for the test data
y_pred_gb = model_gb.predict(X_test_selected_gb)
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
print("Evaluation Results after Feature Selection (RFECV) for Gradient Boosting:")
print(f"  MAE: {mae_gb:.4f}")
print(f"  MSE: {mse_gb:.4f}")
print(f"  RMSE: {rmse_gb:.4f}")
print(f"  R2: {r2_gb:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_gb']]) 
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_gb']])


# In[ ]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
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

# Initialize the Gradient Boosting model with fixed hyperparameters
model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)  # Set hyperparameters as desired

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_gb = RFECV(estimator=model_gb, cv=5, scoring='neg_mean_squared_error')
rfecv_gb.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_gb = rfecv_gb.support_

# Filter the training and test data with selected features
X_train_selected_gb = X_train.iloc[:, selected_features_indices_gb]
X_test_selected_gb = X_test.iloc[:, selected_features_indices_gb]

# Train the Gradient Boosting model with selected features
model_gb.fit(X_train_selected_gb, y_train)

# Predict player ratings for the test data
y_pred_gb = model_gb.predict(X_test_selected_gb)
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
print("Evaluation Results after Feature Selection (RFECV) for Gradient Boosting:")
print(f"  MAE: {mae_gb:.4f}")
print(f"  MSE: {mse_gb:.4f}")
print(f"  RMSE: {rmse_gb:.4f}")
print(f"  R2: {r2_gb:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_gb']]) 
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_gb']])


# In[ ]:


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
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

# Initialize the Ridge Regression model with fixed hyperparameters
model_ridge = Ridge(alpha=1.0, random_state=42)  # Set hyperparameters as desired

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_ridge = RFECV(estimator=model_ridge, cv=5, scoring='neg_mean_squared_error')
rfecv_ridge.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_ridge = rfecv_ridge.support_

# Filter the training and test data with selected features
X_train_selected_ridge = X_train.iloc[:, selected_features_indices_ridge]
X_test_selected_ridge = X_test.iloc[:, selected_features_indices_ridge]

# Train the Ridge Regression model with selected features
model_ridge.fit(X_train_selected_ridge, y_train)

# Predict player ratings for the test data
y_pred_ridge = model_ridge.predict(X_test_selected_ridge)
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
print("Evaluation Results after Feature Selection (RFECV) for Ridge Regression:")
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

# Initialize the Lasso Regression model with fixed hyperparameters
model_lasso = Lasso(alpha=1.0, random_state=42)  # Set hyperparameters as desired

# Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features
rfecv_lasso = RFECV(estimator=model_lasso, cv=5, scoring='neg_mean_squared_error')
rfecv_lasso.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_lasso = rfecv_lasso.support_

# Filter the training and test data with selected features
X_train_selected_lasso = X_train.iloc[:, selected_features_indices_lasso]
X_test_selected_lasso = X_test.iloc[:, selected_features_indices_lasso]

# Train the Lasso Regression model with selected features
model_lasso.fit(X_train_selected_lasso, y_train)

# Predict player ratings for the test data
y_pred_lasso = model_lasso.predict(X_test_selected_lasso)
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
print("Evaluation Results after Feature Selection (RFECV) for Lasso Regression:")
print(f"  MAE: {mae_lasso:.4f}")
print(f"  MSE: {mse_lasso:.4f}")
print(f"  RMSE: {rmse_lasso:.4f}")
print(f"  R2: {r2_lasso:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_lasso']]) 
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_lasso']])




# In[ ]:




