#import xarray as xr
#import rioxarray as rxr
import os
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
import joblib
import time


import pandas as pd
import pandas as pd

# Load your dataframe (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER1_TEST_cities.csv')
# Calculate the size of 10% of the dataframe
sample_size = int(0.05 * len(df))
# Take three random subsamples
subsample_1 = df.sample(n=sample_size, replace=False)
subsample_1.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/TEST_subsample_CL1.csv', index=False)



# Load your dataframe (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER2_TEST_cities.csv')
# Calculate the size of 10% of the dataframe
sample_size = int(0.05 * len(df))
# Take three random subsamples
subsample_1 = df.sample(n=sample_size, replace=False)
subsample_1.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/TEST_subsample_CL2.csv', index=False)


# Load your dataframe (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER3_TEST_cities.csv')
# Calculate the size of 10% of the dataframe
sample_size = int(0.05 * len(df))
# Take three random subsamples
subsample_1 = df.sample(n=sample_size, replace=False)
subsample_1.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/TEST_subsample_CL3.csv', index=False)
