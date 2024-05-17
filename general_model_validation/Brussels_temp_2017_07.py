import pandas as pd
import matplotlib.pyplot as plt
import time

# Load data for Brussels
test = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/CITIES3/ready_for_model_Brussels_2017_07.csv')



# Count NaN values per column before dropping
test_nan_before_drop = test.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])

test = test.rename(columns={'T_2M': 'T2M_NC'})
test = test.rename(columns={'T_2M_COR': 'T2M'})

test['T2M_difference'] = test['T_TARGET'] - test['T2M']
test['T2M'] = test['T2M'] - 273.15
test['T2M_NC'] = test['T2M_NC'] - 273.15

test = test.dropna(subset=['T2M_difference'])

# Count NaN values per column after dropping 'T_TARGET'
test_nan_after_drop = test.isnull().sum()

# Fill NaN values with 0 in 'CBH' column

# Count NaN values per column after filling with 0
test_nan_after_fill = test.isnull().sum()

start_time = time.perf_counter()
# Fill NaN values with the median value of the specific 'City'
for column in test.columns:
    if test[column].isnull().any():
        test[column] = test.groupby('city')[column].transform(lambda x: x.fillna(x.median()))


end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Write results to file
with open('run_time.txt', 'a') as f:
    f.write(f"Elapsed time impute:  {elapsed_time} seconds  \n")




# Count NaN values per column after filling with median
test_nan_after_median_fill = test.isnull().sum()

# Calculate number of NaN values filled and dropped per column
test_nan_filled = test_nan_before_drop - test_nan_after_fill

test_nan_dropped = test_nan_after_drop - test_nan_after_median_fill

# Writing results to a file
with open('run_time.txt', 'a') as f:
    f.write("NaN values filled per column before dropping 'T_TARGET':\n")
    f.write("Test Data:\n")
    f.write(test_nan_filled.to_string() + "\n")
    f.write("\n")
    f.write("NaN values dropped per column after dropping 'T_TARGET':\n")
    f.write("Test Data:\n")
    f.write(test_nan_dropped.to_string() + "\n")


y_test = test['T2M_difference'] 
X_test = test[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
import matplotlib.pyplot as plt
import pandas as pd
import joblib
# Load the model using the formatted file path
model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")

# Make predictions
test['T2M_diff_pred']=model.predict(X_test)
test['T2M_diff_UrbClim']=test['T2M_difference']
test['T2M_pred'] = model.predict(X_test)+X_test['T2M']
test['T2M_UrbClim']=test['T2M_difference'] +X_test['T2M']
test['T2M_ERA5']=X_test['T2M']


df_Brussels=test
# Convert 'time' column to datetime if it's not already
df_Brussels['time'] = pd.to_datetime(df_Brussels['time'])



# Group by 'time' and calculate the mean temperature for each time point
mean_T2M_diff_pred_Brussels = df_Brussels.groupby('time')['T2M_diff_pred'].mean()

# Group by 'time' and calculate the mean temperature for each time point
mean_T2M_diff_UrbClim_Brussels = df_Brussels.groupby('time')['T2M_diff_UrbClim'].mean()



# Group by 'time' and calculate the mean temperature for each time point
mean_T2M_pred_Brussels = df_Brussels.groupby('time')['T2M_pred'].mean()

# Group by 'time' and calculate the mean temperature for each time point
mean_T2M_UrbClim_Brussels = df_Brussels.groupby('time')['T2M_UrbClim'].mean()

mean_T2M_ERA5_Brussels = df_Brussels.groupby('time')['T2M_ERA5'].mean()


mean_T2M_pred_Brussels.to_csv('mean_T2M_pred_Brussels07.csv', header=True)

mean_T2M_UrbClim_Brussels.to_csv('mean_T2M_UrbClim_Brussels07.csv', header=True)

mean_T2M_diff_pred_Brussels.to_csv('mean_T2M_diff_pred_Brussels07.csv', header=True)

mean_T2M_diff_UrbClim_Brussels.to_csv('mean_mean_T2M_diff_UrbClim_Brussels07.csv', header=True)

mean_T2M_ERA5_Brussels.to_csv('mean_T2M_ERA5_Brussels07.csv', header=True)
