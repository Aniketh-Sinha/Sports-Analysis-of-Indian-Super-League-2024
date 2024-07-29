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


# In[6]:


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

# Initialize the Random Forest model with different hyperparameters
model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)

# Train the Random Forest model
model.fit(X_train, y_train)

# Predict player ratings for the test data
y_pred = model.predict(X_test)
test_data.loc[:, 'Predicted_Rating'] = y_pred 

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print("Evaluation Results:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating']])


# In[7]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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

# Initialize the Gradient Boosting model with different hyperparameters
model_gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)  

# Train the Gradient Boosting model
model_gb.fit(X_train, y_train)

# Predict player ratings for the test data using Gradient Boosting
y_pred_gb = model_gb.predict(X_test)
test_data.loc[:, 'Predicted_Rating_gb'] = y_pred_gb  

# Calculate evaluation metrics for Gradient Boosting
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Display evaluation results for Gradient Boosting
print("Evaluation Results for Gradient Boosting:")
print(f"  MAE: {mae_gb:.4f}")
print(f"  MSE: {mse_gb:.4f}")
print(f"  RMSE: {rmse_gb:.4f}")
print(f"  R2: {r2_gb:.4f}")

# Rank players based on actual ratings for Gradient Boosting
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings for Gradient Boosting
test_data = test_data.sort_values(by='Predicted_Rating_gb', ascending=False)
test_data['Predicted_Rank_gb'] = range(1, len(test_data) + 1)

# Display evaluation results
print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_gb']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_gb']])



# In[8]:


import pandas as pd
from xgboost import XGBRegressor
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

# Initialize the XGBoost model with different hyperparameters
model_xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)  

# Train the XGBoost model
model_xgb.fit(X_train, y_train)

# Predict player ratings for the test data using XGBoost
y_pred_xgb = model_xgb.predict(X_test)
test_data.loc[:, 'Predicted_Rating_xgb'] = y_pred_xgb  

# Calculate evaluation metrics for XGBoost
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Display evaluation results for XGBoost
print("Evaluation Results for XGBoost:")
print(f"  MAE: {mae_xgb:.4f}")
print(f"  MSE: {mse_xgb:.4f}")
print(f"  RMSE: {rmse_xgb:.4f}")
print(f"  R2: {r2_xgb:.4f}")

# Rank players based on actual ratings for XGBoost
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings for XGBoost
test_data = test_data.sort_values(by='Predicted_Rating_xgb', ascending=False)
test_data['Predicted_Rank_xgb'] = range(1, len(test_data) + 1)

# Display evaluation results
print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_xgb']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_xgb']])




# In[9]:


import pandas as pd
from sklearn.linear_model import Ridge
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

# Initialize the Ridge Regression model with fixed hyperparameters
model_ridge = Ridge(alpha=1.0, random_state=42)  # Set hyperparameters as desired

# Train the Ridge Regression model
model_ridge.fit(X_train, y_train)

# Predict player ratings for the test data using Ridge Regression
y_pred_ridge = model_ridge.predict(X_test)
test_data.loc[:, 'Predicted_Rating_ridge'] = y_pred_ridge  

# Calculate evaluation metrics for Ridge Regression
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Display evaluation results for Ridge Regression
print("Evaluation Results for Ridge Regression:")
print(f"  MAE: {mae_ridge:.4f}")
print(f"  MSE: {mse_ridge:.4f}")
print(f"  RMSE: {rmse_ridge:.4f}")
print(f"  R2: {r2_ridge:.4f}")

# Rank players based on actual ratings for Ridge Regression
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings for Ridge Regression
test_data = test_data.sort_values(by='Predicted_Rating_ridge', ascending=False)
test_data['Predicted_Rank_ridge'] = range(1, len(test_data) + 1)

# Display evaluation results
print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_ridge']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_ridge']])




# In[11]:


import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
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

# Perform feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Lasso Regression model with cross-validation for alpha selection
model_lasso_cv = LassoCV(cv=5, random_state=42)

# Train the Lasso Regression model with cross-validation
model_lasso_cv.fit(X_train_scaled, y_train)

# Get the optimal alpha value selected by cross-validation
optimal_alpha = model_lasso_cv.alpha_

# Train the Lasso Regression model with the optimal alpha
model_lasso_final = Lasso(alpha=optimal_alpha, random_state=42)
model_lasso_final.fit(X_train_scaled, y_train)

# Predict player ratings for the test data using the final model
y_pred_lasso_final = model_lasso_final.predict(X_test_scaled)
test_data.loc[:, 'Predicted_Rating_lasso_final'] = y_pred_lasso_final  

# Calculate evaluation metrics for Lasso Regression with the final model
mae_lasso_final = mean_absolute_error(y_test, y_pred_lasso_final)
mse_lasso_final = mean_squared_error(y_test, y_pred_lasso_final)
rmse_lasso_final = sqrt(mse_lasso_final)
r2_lasso_final = r2_score(y_test, y_pred_lasso_final)

# Display evaluation results for Lasso Regression with the final model
print("Evaluation Results for Lasso Regression with Final Model:")
print(f"  MAE: {mae_lasso_final:.4f}")
print(f"  MSE: {mse_lasso_final:.4f}")
print(f"  RMSE: {rmse_lasso_final:.4f}")
print(f"  R2: {r2_lasso_final:.4f}")

# Rank players based on actual ratings for Lasso Regression
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings for Lasso Regression
test_data = test_data.sort_values(by='Predicted_Rating_lasso_final', ascending=False)
test_data['Predicted_Rank_lasso_final'] = range(1, len(test_data) + 1)

# Display evaluation results and rankings
print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank_lasso_final']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating_lasso_final']])





# In[ ]:




