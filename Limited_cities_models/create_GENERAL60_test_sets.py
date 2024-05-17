
import pandas as pd
import pandas as pd




# Load your dataframe (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv')

data = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_test_FINAL.csv')
# Filter the DataFrame to select only the cities where Cluster is 1
filtered_data = data[data['Cluster'] == 0]
# Select only the 'City' column from the filtered DataFrame
selected_cities1 = filtered_data['City']
# Print the selected cities
print(selected_cities1)

# Filter the DataFrame to select only the cities where Cluster is 1
filtered_data = data[data['Cluster'] == 1]
# Select only the 'City' column from the filtered DataFrame
selected_cities2 = filtered_data['City']
# Print the selected cities
print(selected_cities2)

# Filter the DataFrame to select only the cities where Cluster is 1
filtered_data = data[data['Cluster'] == 2]
# Select only the 'City' column from the filtered DataFrame
selected_cities3 = filtered_data['City']
# Print the selected cities
print(selected_cities3)


# Save subsamples for all cities
subsample = df[df['city'].isin(selected_cities1)]
subsample.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER1_TEST_cities.csv', index=False)


# Save subsamples for all cities
subsample = df[df['city'].isin(selected_cities2)]
subsample.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER2_TEST_cities.csv', index=False)

# Save subsamples for all cities
subsample = df[df['city'].isin(selected_cities3)]
subsample.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER3_TEST_cities.csv', index=False)
