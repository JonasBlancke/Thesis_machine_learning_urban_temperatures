import xarray as xr
#import rioxarray as rxr
import os
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.express as px
import time
import seaborn as sns
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt







file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_data_FINAL.csv'
df = pd.read_csv(file_path, usecols=['CBH','T_2M_COR', 'T_SK', 'T_TARGET', 'city', 'hour', 'LC_CORINE', 'LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF'])


df = df.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
train=df
train['T2M_difference']=train['T_TARGET'] - train['T2M']
train['T2M']=train['T2M']-273.15
train['SKT']=train['SKT']-273.15
train['CBH'] = train['CBH'].fillna(0)
nan_values_per_column = train.isnull().sum()
print("NaN values per column:")
print(nan_values_per_column)
# Drop all rows with NaN values in any column
train = train.dropna()


file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_train_FINAL.csv'
clusterval=pd.read_csv(file_path)
print(clusterval)

cluster1_cities= clusterval[clusterval['Cluster'] == 0]['City']
cluster2_cities= clusterval[clusterval['Cluster'] == 1]['City']
cluster3_cities= clusterval[clusterval['Cluster'] == 2]['City']
print(cluster1_cities)





cluster1 = train[train['city'].isin(cluster1_cities)]
cluster2 = train[train['city'].isin(cluster2_cities)]
cluster3 = train[train['city'].isin(cluster3_cities)]


#spatial plot:




# Calculate correlations with 'T2M_difference' for each cluster
correlation_cluster1 = cluster1.corr()['T2M_difference'].dropna()
correlation_cluster2 = cluster2.corr()['T2M_difference'].dropna()
correlation_cluster3 = cluster3.corr()['T2M_difference'].dropna()
# Extract variables of interest

import numpy as np
import matplotlib.pyplot as plt

# Extract variables of interest
variables_of_interest = ['LC_CORINE', 'LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF']

# Calculate the positions for each variable
bar_width = 0.25  # Width of each bar
x = np.arange(len(variables_of_interest))  # Generate x values for the bars

# Plot bars for Cluster 1 in blue
plt.figure(figsize=(15, 8))
plt.bar(x - bar_width, correlation_cluster1[variables_of_interest], color='blue', width=bar_width, label='Cluster 1')

# Plot bars for Cluster 2 in green
plt.bar(x, correlation_cluster2[variables_of_interest], color='green', width=bar_width, label='Cluster 2')

# Plot bars for Cluster 3 in red
plt.bar(x + bar_width, correlation_cluster3[variables_of_interest], color='red', width=bar_width, label='Cluster 3')

# Set title and labels with larger font size
plt.title('Correlation of the spatial candidate features with the target variable: T2M_difference', fontsize=22)
plt.ylabel('Correlation', fontsize=18)
plt.xticks(ticks=x, labels=variables_of_interest, rotation=45, ha='right', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('correlation_with_t2m_difference_spatial.png', format='png')
plt.show()

