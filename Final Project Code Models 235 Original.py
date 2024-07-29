#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# In[10]:


df5.columns


# In[11]:


# Filter data for the year 2023
df_2023 = df5[df5['Year'] == 2023].copy()


# In[12]:


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

# Initialize the Random Forest model with fixed hyperparameters
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # Set hyperparameters as desired

# Train the Random Forest model
model.fit(X_train, y_train)

# Predict player ratings for the test data
y_pred = model.predict(X_test)
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
print("Evaluation Results:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating']])

import matplotlib.pyplot as plt
import numpy as np

# Assuming test_data is your DataFrame containing PlayerRating and PredictedRating
# Sort test_data by PlayerRating to get the top 10 players
top_10_players = test_data.sort_values(by=['Predicted_Rating'], ascending=False).head(10)

# Getting the indices
indices = np.arange(len(top_10_players))

# Setting figure size
plt.figure(figsize=(12, 8))

# Define the width of the bars
bar_width = 0.35

# Plotting the clustered bar graph
plt.bar(indices - bar_width/2, top_10_players['PlayerRating'], width=bar_width, label='Player Rating', color='skyblue', alpha=0.7)
plt.bar(indices + bar_width/2, top_10_players['Predicted_Rating'], width=bar_width, label='Predicted Rating', color='orange', alpha=0.7)

# Adding labels and title with larger font sizes
plt.xlabel('Players', fontsize=14)
plt.ylabel('Rating Value', fontsize=14)
plt.title('Comparison of Player Rating and Predicted Rating (Top 10 Players)', fontsize=16)

# Adding legend with larger font size
plt.legend(fontsize=12)

# Setting xticks with larger font size and equal spacing
plt.xticks(indices, top_10_players['Player'], rotation=45, fontsize=12)

# Setting yticks with larger font size
plt.yticks(fontsize=12)

# Adding grid lines
plt.grid(True)

# Displaying the graph
plt.tight_layout()  # Adjust layout to make everything fit without overlap
plt.show()


# In[13]:


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

# Initialize the Gradient Boosting model with fixed hyperparameters
model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)  # Set hyperparameters as desired

# Train the Gradient Boosting model
model.fit(X_train, y_train)

# Predict player ratings for the test data
y_pred = model.predict(X_test)
test_data.loc[:, 'Predicted_Rating1'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating1', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating1']])

import matplotlib.pyplot as plt
import numpy as np

# Assuming test_data is your DataFrame containing PlayerRating and PredictedRating
# Sort test_data by PlayerRating to get the top 10 players
top_10_players = test_data.sort_values(by=['Predicted_Rating1'], ascending=False).head(10)

# Getting the indices
indices = np.arange(len(top_10_players))

# Setting figure size
plt.figure(figsize=(12, 8))

# Define the width of the bars
bar_width = 0.35

# Plotting the clustered bar graph
plt.bar(indices - bar_width/2, top_10_players['PlayerRating'], width=bar_width, label='Player Rating', color='skyblue', alpha=0.7)
plt.bar(indices + bar_width/2, top_10_players['Predicted_Rating1'], width=bar_width, label='Predicted Rating', color='orange', alpha=0.7)

# Adding labels and title with larger font sizes
plt.xlabel('Players', fontsize=14)
plt.ylabel('Rating Value', fontsize=14)
plt.title('Comparison of Player Rating and Predicted Rating (Top 10 Players)', fontsize=16)

# Adding legend with larger font size
plt.legend(fontsize=12)

# Setting xticks with larger font size and equal spacing
plt.xticks(indices, top_10_players['Player'], rotation=45, fontsize=12)

# Setting yticks with larger font size
plt.yticks(fontsize=12)

# Adding grid lines
plt.grid(True)

# Displaying the graph
plt.tight_layout()  # Adjust layout to make everything fit without overlap
plt.show()



# In[14]:


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

# Initialize the XGBoost model with fixed hyperparameters
model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)  # Set hyperparameters as desired

# Train the XGBoost model
model.fit(X_train, y_train)

# Predict player ratings for the test data
y_pred = model.predict(X_test)
test_data.loc[:, 'Predicted_Rating2'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating2', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating2']])

import matplotlib.pyplot as plt
import numpy as np

# Assuming test_data is your DataFrame containing PlayerRating and PredictedRating
# Sort test_data by PlayerRating to get the top 10 players
top_10_players = test_data.sort_values(by=['Predicted_Rating2'], ascending=False).head(10)

# Getting the indices
indices = np.arange(len(top_10_players))

# Setting figure size
plt.figure(figsize=(12, 8))

# Define the width of the bars
bar_width = 0.35

# Plotting the clustered bar graph
plt.bar(indices - bar_width/2, top_10_players['PlayerRating'], width=bar_width, label='Player Rating', color='skyblue', alpha=0.7)
plt.bar(indices + bar_width/2, top_10_players['Predicted_Rating2'], width=bar_width, label='Predicted Rating', color='orange', alpha=0.7)

# Adding labels and title with larger font sizes
plt.xlabel('Players', fontsize=14)
plt.ylabel('Rating Value', fontsize=14)
plt.title('Comparison of Player Rating and Predicted Rating (Top 10 Players)', fontsize=16)

# Adding legend with larger font size
plt.legend(fontsize=12)

# Setting xticks with larger font size and equal spacing
plt.xticks(indices, top_10_players['Player'], rotation=45, fontsize=12)

# Setting yticks with larger font size
plt.yticks(fontsize=12)

# Adding grid lines
plt.grid(True)

# Displaying the graph
plt.tight_layout()  # Adjust layout to make everything fit without overlap
plt.show()



# In[15]:


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
model = Ridge(alpha=1.0, random_state=42)  # Set hyperparameters as desired

# Train the Ridge Regression model
model.fit(X_train, y_train)

# Predict player ratings for the test data
y_pred = model.predict(X_test)
test_data.loc[:, 'Predicted_Rating3'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating3', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating3']])

import matplotlib.pyplot as plt
import numpy as np

# Assuming test_data is your DataFrame containing PlayerRating and PredictedRating
# Sort test_data by PlayerRating to get the top 10 players
top_10_players = test_data.sort_values(by=['Predicted_Rating3'], ascending=False).head(10)

# Getting the indices
indices = np.arange(len(top_10_players))

# Setting figure size
plt.figure(figsize=(12, 8))

# Define the width of the bars
bar_width = 0.35

# Plotting the clustered bar graph
plt.bar(indices - bar_width/2, top_10_players['PlayerRating'], width=bar_width, label='Player Rating', color='skyblue', alpha=0.7)
plt.bar(indices + bar_width/2, top_10_players['Predicted_Rating3'], width=bar_width, label='Predicted Rating', color='orange', alpha=0.7)

# Adding labels and title with larger font sizes
plt.xlabel('Players', fontsize=14)
plt.ylabel('Rating Value', fontsize=14)
plt.title('Comparison of Player Rating and Predicted Rating (Top 10 Players)', fontsize=16)

# Adding legend with larger font size
plt.legend(fontsize=12)

# Setting xticks with larger font size and equal spacing
plt.xticks(indices, top_10_players['Player'], rotation=45, fontsize=12)

# Setting yticks with larger font size
plt.yticks(fontsize=12)

# Adding grid lines
plt.grid(True)

# Displaying the graph
plt.tight_layout()  # Adjust layout to make everything fit without overlap
plt.show()




# In[16]:


import pandas as pd
from sklearn.linear_model import Lasso
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

# Initialize the Lasso Regression model with fixed hyperparameters
model = Lasso(alpha=1.0, random_state=42)  # Set hyperparameters as desired

# Train the Lasso Regression model
model.fit(X_train, y_train)

# Predict player ratings for the test data
y_pred = model.predict(X_test)
test_data.loc[:, 'Predicted_Rating4'] = y_pred  # Use .loc to set values

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Rank players based on actual ratings
test_data = test_data.sort_values(by='PlayerRating', ascending=False)
test_data['Actual_Rank'] = range(1, len(test_data) + 1)

# Rank players based on predicted ratings
test_data = test_data.sort_values(by='Predicted_Rating4', ascending=False)
test_data['Predicted_Rank'] = range(1, len(test_data) + 1)

# Display evaluation results
print("Evaluation Results:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2: {r2:.4f}")

print(test_data[['Player', 'Pos', 'Actual_Rank', 'Predicted_Rank']])
print(test_data[['Player', 'Pos','PlayerRating', 'Predicted_Rating4']])

import matplotlib.pyplot as plt
import numpy as np

# Assuming test_data is your DataFrame containing PlayerRating and PredictedRating
# Sort test_data by PlayerRating to get the top 10 players
top_10_players = test_data.sort_values(by=['Predicted_Rating4'], ascending=False).head(10)

# Getting the indices
indices = np.arange(len(top_10_players))

# Setting figure size
plt.figure(figsize=(12, 8))

# Define the width of the bars
bar_width = 0.35

# Plotting the clustered bar graph
plt.bar(indices - bar_width/2, top_10_players['PlayerRating'], width=bar_width, label='Player Rating', color='skyblue', alpha=0.7)
plt.bar(indices + bar_width/2, top_10_players['Predicted_Rating4'], width=bar_width, label='Predicted Rating', color='orange', alpha=0.7)

# Adding labels and title with larger font sizes
plt.xlabel('Players', fontsize=14)
plt.ylabel('Rating Value', fontsize=14)
plt.title('Comparison of Player Rating and Predicted Rating (Top 10 Players)', fontsize=16)

# Adding legend with larger font size
plt.legend(fontsize=12)

# Setting xticks with larger font size and equal spacing
plt.xticks(indices, top_10_players['Player'], rotation=45, fontsize=12)

# Setting yticks with larger font size
plt.yticks(fontsize=12)

# Adding grid lines
plt.grid(True)

# Displaying the graph
plt.tight_layout()  # Adjust layout to make everything fit without overlap
plt.show()



