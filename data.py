import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

data_1 = pd.read_csv('data.csv')
print(data_1)
print(data_1.isna().any().any())

df = data_1.copy()

df['time'] = pd.to_datetime(df['time'])

# Extract the hour of the day from the 'time' column
df['hour'] = df['time'].dt.hour

# Encode 'location' with numerical IDs
location_ids = {location: i for i, location in enumerate(df['location'].unique())}
df['location_id'] = df['location'].map(location_ids)

# Create a correlation matrix
corr_matrix = df[['consumption', 'temperature', 'hour', 'location_id']].corr()

# Create a pairplot to visualize relationships between variables
sns.pairplot(df[['consumption', 'temperature', 'hour']])
plt.show()

# Create a correlation heatmap to see correlation coefficients
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Create a categorical scatter plot for location vs. consumption
sns.catplot(x='location', y='consumption', data=df, kind='boxen', height=8, aspect=2)
plt.xticks(rotation=45)
plt.title('Consumption by Location')
plt.show()

# Create a scatter plot for hour vs. consumption
sns.scatterplot(x='hour', y='consumption', data=df)
plt.title('Hour of the Day vs. Consumption')
plt.show()

unique_locations = data_1['location'].unique()

data_1['hour'] = pd.to_datetime(data_1['time']).dt.hour
data_1['day_of_week'] = pd.to_datetime(data_1['time']).dt.dayofweek
data_1['month'] = pd.to_datetime(data_1['time']).dt.month

location_dataframes = {}
for location in unique_locations:
    location_dataframes[location] = data_1[data_1['location'] == location]

oslo_data = location_dataframes['oslo']
bergen_data = location_dataframes['bergen']
stavanger_data = location_dataframes['stavanger']
trondheim_data = location_dataframes['trondheim']
tromsø_data = location_dataframes['tromsø']
helsingfors_data = location_dataframes['helsingfors']