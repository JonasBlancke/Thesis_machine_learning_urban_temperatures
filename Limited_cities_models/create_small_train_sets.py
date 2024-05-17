
import pandas as pd
import pandas as pd


# Load your dataframe (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_data_FINAL.csv')

# Define cities for each cluster
cluster_cities = {
    'CLUSTER1': ['Dublin', 'London', 'Stockholm', 'Lille', 'Tartu'],
    'CLUSTER2': ['Malaga', 'Palermo', 'Nice', 'Split', 'Bordeaux'],
    'CLUSTER3': ['Pecs', 'Wroclaw', 'Toulouse', 'Bucharest', 'Strasbourg']
}

# Save subsamples for each cluster
for cluster, cities in cluster_cities.items():
    subsample = df[df['city'].isin(cities)]
    subsample.to_csv(f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/{cluster}_5cities.csv', index=False)

# Save small subsamples for each cluster
small_clusters = {
    'CLUSTER1': ['Stockholm', 'Lille'],
    'CLUSTER2': ['Malaga', 'Split'],
    'CLUSTER3': ['Strasbourg', 'Pecs']
}

for cluster, cities in small_clusters.items():
    subsample = df[df['city'].isin(cities)]
    subsample.to_csv(f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/{cluster}_2cities.csv', index=False)
