# -*- coding: utf-8 -*-
"""Kopie van plot_cities.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tevFlyMwlaokt7TKZ4_AREEGMeJiqqOg
"""

!pip install cartopy
!pip install mpl_toolkits

!pip install matplotlib-scalebar

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv')

# Create a map of Europe in EPSG:4326
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set extent to Europe
ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

# Add natural earth features for coastline and countries
ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

# Get unique clusters
clusters = df['Cluster'].unique()

# Define colors for clusters
cluster_colors = ['blue', 'green', 'red']

# Sort clusters
clusters = sorted(clusters)

legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
legend_handles = []

for i, cluster in enumerate(clusters):
    cluster_data = df[df['Cluster'] == cluster]
    color = cluster_colors[i]
    handle, = ax.plot(cluster_data['Longitude'], cluster_data['Latitude'], 'o', color=color, markersize=5, label=legend_labels[i], transform=ccrs.PlateCarree())
    legend_handles.append(handle)

# Add legend with handles in reversed order to arrange labels from top to bottom
plt.legend(handles=legend_handles)

# Add gridlines with labels only on the bottom and left sides
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlines = True
gl.ylines = True
gl.xlabels_bottom = True
gl.ylabels_left = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Set title

plt.show()

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd

# Read the CSV file
df = pd.read_csv('/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv')

# Define colors for clusters
cluster_colors = ['blue', 'green', 'red']

# Define legend labels
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# Iterate over each function type ('T', 'V', 'R')
for function_type in ['T', 'V', 'R']:
    function_data = df[df['function'] == function_type]

    # Create separate figure for each function type
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set extent to Europe for each subplot
    ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

    # Add natural earth features for coastline and countries for each subplot
    ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

    # Get unique clusters
    clusters = function_data['Cluster'].unique()

    # Sort clusters
    clusters = sorted(clusters)

    legend_handles = []

    # Plot cities for each cluster
    for j, cluster in enumerate(clusters):
        cluster_data = function_data[function_data['Cluster'] == cluster]
        color = cluster_colors[j]
        handle = ax.plot(cluster_data['Longitude'], cluster_data['Latitude'], 'o', color=color, markersize=5, label=legend_labels[j], transform=ccrs.PlateCarree())
        legend_handles.append(handle[0])

    # Add legend with handles for each subplot
    ax.legend(handles=legend_handles, loc='upper right')

    # Add gridlines with labels only on the bottom and left sides for each subplot
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlines = True
    gl.ylines = True
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add title to each subplot
    if function_type == 'T':
        ax.set_title('Train cities')
    elif function_type == 'V':
        ax.set_title('Validation cities')
    elif function_type == 'R':
        ax.set_title('Test cities')

    plt.show()

"""## **MEAN STATISTICS**"""

!pip install adjustText

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from adjustText import adjust_text
import matplotlib.patches as patches

# Read the CSV files
validation_path = '/content/drive/MyDrive/thesis/MEANSTAT/general_difference_per_city_validation.csv'
validation_values = pd.read_csv(validation_path)
train_path = '/content/drive/MyDrive/thesis/MEANSTAT/general_difference_per_city_train.csv'
train_values = pd.read_csv(train_path)
test_path = '/content/drive/MyDrive/thesis/MEANSTAT/general_difference_per_city_test.csv'
test_values = pd.read_csv(test_path)

# Read the cluster results CSV file
cluster_path = '/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data.rename(columns={'City': 'city'}, inplace=True)

# Define city groups
city_groups = {'Amsterdam': ['Rotterdam', 'Amsterdam', 'Utrecht'], 'Bratislava': ['Vienna', 'Bratislava', 'Gyor']}

# Calculate mean for Rotterdam group and assign to Rotterdam
Amsterdam_mean = train_values.loc[train_values['city'].isin(['Rotterdam', 'Amsterdam', 'Utrecht']), '0'].mean()
train_values.loc[train_values['city'] == 'Amsterdam', '0'] = Amsterdam_mean

# Calculate mean for Graz group and assign to Graz
Bratislava_mean = train_values.loc[train_values['city'].isin(['Vienna', 'Bratislava', 'Gyor']), '0'].mean()
train_values.loc[train_values['city'] == 'Bratislava', '0'] = Bratislava_mean

# Add a new column containing the first 4 letters of the city name
cluster_data['city_short'] = cluster_data['city'].str[:4]
test_values['city_short'] = test_values['city'].str[:4]
train_values['city_short'] = train_values['city'].str[:4]
validation_values['city_short'] = validation_values['city'].str[:4]

# Merge based on the 'city_short' column
test_merged = pd.merge(cluster_data, test_values, on='city_short')
train_merged = pd.merge(cluster_data, train_values, on='city_short')
validation_merged = pd.merge(cluster_data, validation_values, on='city_short')

# Define colors for clusters
cluster_colors = ['blue', 'green', 'red']

# Define legend labels
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# Function to check if two points are close to each other
def are_close(x1, y1, x2, y2, threshold=1):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold

# Preprocess data to identify clusters of close cities
def preprocess_data(df):
    tree = KDTree(df[['Longitude', 'Latitude']])
    close_points = tree.query_ball_point(df[['Longitude', 'Latitude']], r=1)
    df['close_points'] = close_points

# Preprocess data for each merged dataframe
preprocess_data(train_merged)
preprocess_data(validation_merged)
preprocess_data(test_merged)

# Iterate over each function type ('T', 'V', 'R')
for function_type, merged_df in zip(['T', 'V', 'R'], [train_merged, validation_merged, test_merged]):
    # Create separate figure for each function type
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})  # Increase figure size

    # Set extent to Europe for each subplot
    ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

    # Add natural earth features for coastline and countries for each subplot
    ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

    # Get unique clusters
    clusters = merged_df['Cluster'].unique()

    # Sort clusters
    clusters = sorted(clusters)

    legend_handles = []

    # Plot cities for each cluster
    for j, cluster in enumerate(clusters):
        cluster_data = merged_df[merged_df['Cluster'] == cluster]
        color = cluster_colors[j]
        handle = ax.plot(cluster_data['Longitude'], cluster_data['Latitude'], 'o', color=color, markersize=10, label=legend_labels[j], transform=ccrs.PlateCarree())
        legend_handles.append(handle[0])

    # Add legend with handles for each subplot
    ax.legend(handles=legend_handles, loc='upper right')

    # Add gridlines with labels only on the bottom and left sides for each subplot
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlines = True
    gl.ylines = True
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add title to each subplot
    if function_type == 'T':
        ax.set_title('Train cities')
    elif function_type == 'V':
        ax.set_title('Validation cities')
    elif function_type == 'R':
        ax.set_title('Test cities')

    # Annotate points with values, adjusting for overlap
    texts = []
    for i, row in merged_df.iterrows():
        city = row['city_x']
        value = row['0']
        x, y = row['Longitude'], row['Latitude']

        # Adjust text position for cities close to each other
        close_points = row['close_points']
        if city in ['Rotterdam', 'Utrecht', 'Vienna', 'Gyor']:
            continue

        # Create white box behind text
        rect = patches.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='white', alpha=0.5)
        ax.add_patch(rect)

        # Annotate the point with black text and transparent white background box
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)  # Define transparent textbox properties
        texts.append(ax.text(x, y, f'{value:.2f}', fontsize=10, color='black', fontweight='bold', ha='center', va='bottom', bbox=bbox_props))

    # Adjust text to avoid overlap
    adjust_text(texts, ax=ax)

plt.show()

"""# **UHI intensity**"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from adjustText import adjust_text
import matplotlib.patches as patches

# Read the CSV files
validation_path = '/content/drive/MyDrive/thesis/MEANSTAT/UHII_difference_per_city_validation.csv'
validation_values = pd.read_csv(validation_path)
train_path = '/content/drive/MyDrive/thesis/MEANSTAT/UHII_difference_per_city_train.csv'
train_values = pd.read_csv(train_path)
test_path = '/content/drive/MyDrive/thesis/MEANSTAT/UHII_difference_per_city_test.csv'
test_values = pd.read_csv(test_path)

# Read the cluster results CSV file
cluster_path = '/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data.rename(columns={'City': 'city'}, inplace=True)

# Define city groups
city_groups = {'Amsterdam': ['Rotterdam', 'Amsterdam', 'Utrecht'], 'Bratislava': ['Vienna', 'Bratislava', 'Gyor']}

# Calculate mean for Rotterdam group and assign to Rotterdam
Amsterdam_mean = train_values.loc[train_values['city'].isin(['Rotterdam', 'Amsterdam', 'Utrecht']), '0'].mean()
train_values.loc[train_values['city'] == 'Amsterdam', '0'] = Amsterdam_mean

# Calculate mean for Graz group and assign to Graz
Bratislava_mean = train_values.loc[train_values['city'].isin(['Vienna', 'Bratislava', 'Gyor']), '0'].mean()
train_values.loc[train_values['city'] == 'Bratislava', '0'] = Bratislava_mean

# Add a new column containing the first 4 letters of the city name
cluster_data['city_short'] = cluster_data['city'].str[:4]
test_values['city_short'] = test_values['city'].str[:4]
train_values['city_short'] = train_values['city'].str[:4]
validation_values['city_short'] = validation_values['city'].str[:4]

# Merge based on the 'city_short' column
test_merged = pd.merge(cluster_data, test_values, on='city_short')
train_merged = pd.merge(cluster_data, train_values, on='city_short')
validation_merged = pd.merge(cluster_data, validation_values, on='city_short')

# Define colors for clusters
cluster_colors = ['blue', 'green', 'red']

# Define legend labels
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# Function to check if two points are close to each other
def are_close(x1, y1, x2, y2, threshold=1):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold

# Preprocess data to identify clusters of close cities
def preprocess_data(df):
    tree = KDTree(df[['Longitude', 'Latitude']])
    close_points = tree.query_ball_point(df[['Longitude', 'Latitude']], r=1)
    df['close_points'] = close_points

# Preprocess data for each merged dataframe
preprocess_data(train_merged)
preprocess_data(validation_merged)
preprocess_data(test_merged)

# Iterate over each function type ('T', 'V', 'R')
for function_type, merged_df in zip(['T', 'V', 'R'], [train_merged, validation_merged, test_merged]):
    # Create separate figure for each function type
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})  # Increase figure size

    # Set extent to Europe for each subplot
    ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

    # Add natural earth features for coastline and countries for each subplot
    ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

    # Get unique clusters
    clusters = merged_df['Cluster'].unique()

    # Sort clusters
    clusters = sorted(clusters)

    legend_handles = []

    # Plot cities for each cluster
    for j, cluster in enumerate(clusters):
        cluster_data = merged_df[merged_df['Cluster'] == cluster]
        color = cluster_colors[j]
        handle = ax.plot(cluster_data['Longitude'], cluster_data['Latitude'], 'o', color=color, markersize=10, label=legend_labels[j], transform=ccrs.PlateCarree())
        legend_handles.append(handle[0])

    # Add legend with handles for each subplot
    ax.legend(handles=legend_handles, loc='upper right')

    # Add gridlines with labels only on the bottom and left sides for each subplot
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlines = True
    gl.ylines = True
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add title to each subplot
    if function_type == 'T':
        ax.set_title('Train cities: UHI intensity')
    elif function_type == 'V':
        ax.set_title('Validation cities: UHI intensity')
    elif function_type == 'R':
        ax.set_title('Test cities: UHI intensity')

    # Annotate points with values, adjusting for overlap
    texts = []
    for i, row in merged_df.iterrows():
        city = row['city_x']
        value = row['0']
        x, y = row['Longitude'], row['Latitude']

        # Adjust text position for cities close to each other
        close_points = row['close_points']
        if city in ['Rotterdam', 'Utrecht', 'Vienna', 'Gyor']:
            continue

        # Create white box behind text
        rect = patches.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='white', alpha=0.5)
        ax.add_patch(rect)

        # Annotate the point with black text and transparent white background box
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)  # Define transparent textbox properties
        texts.append(ax.text(x, y, f'{value:.2f}', fontsize=10, color='black', fontweight='bold', ha='center', va='bottom', bbox=bbox_props))

    # Adjust text to avoid overlap
    adjust_text(texts, ax=ax)

plt.show()

"""# **Rural diff**"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from adjustText import adjust_text
import matplotlib.patches as patches

# Read the CSV files
validation_path = '/content/drive/MyDrive/thesis/MEANSTAT/rural_difference_per_city_validation.csv'
validation_values = pd.read_csv(validation_path)
train_path = '/content/drive/MyDrive/thesis/MEANSTAT/rural_difference_per_city_train.csv'
train_values = pd.read_csv(train_path)
test_path = '/content/drive/MyDrive/thesis/MEANSTAT/rural_difference_per_city_test.csv'
test_values = pd.read_csv(test_path)

# Read the cluster results CSV file
cluster_path = '/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data.rename(columns={'City': 'city'}, inplace=True)

# Define city groups
city_groups = {'Amsterdam': ['Rotterdam', 'Amsterdam', 'Utrecht'], 'Bratislava': ['Vienna', 'Bratislava', 'Gyor']}

# Calculate mean for Rotterdam group and assign to Rotterdam
Amsterdam_mean = train_values.loc[train_values['city'].isin(['Rotterdam', 'Amsterdam', 'Utrecht']), '0'].mean()
train_values.loc[train_values['city'] == 'Amsterdam', '0'] = Amsterdam_mean

# Calculate mean for Graz group and assign to Graz
Bratislava_mean = train_values.loc[train_values['city'].isin(['Vienna', 'Bratislava', 'Gyor']), '0'].mean()
train_values.loc[train_values['city'] == 'Bratislava', '0'] = Bratislava_mean

# Add a new column containing the first 4 letters of the city name
cluster_data['city_short'] = cluster_data['city'].str[:4]
test_values['city_short'] = test_values['city'].str[:4]
train_values['city_short'] = train_values['city'].str[:4]
validation_values['city_short'] = validation_values['city'].str[:4]

# Merge based on the 'city_short' column
test_merged = pd.merge(cluster_data, test_values, on='city_short')
train_merged = pd.merge(cluster_data, train_values, on='city_short')
validation_merged = pd.merge(cluster_data, validation_values, on='city_short')

# Define colors for clusters
cluster_colors = ['blue', 'green', 'red']

# Define legend labels
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# Function to check if two points are close to each other
def are_close(x1, y1, x2, y2, threshold=1):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold

# Preprocess data to identify clusters of close cities
def preprocess_data(df):
    tree = KDTree(df[['Longitude', 'Latitude']])
    close_points = tree.query_ball_point(df[['Longitude', 'Latitude']], r=1)
    df['close_points'] = close_points

# Preprocess data for each merged dataframe
preprocess_data(train_merged)
preprocess_data(validation_merged)
preprocess_data(test_merged)

# Iterate over each function type ('T', 'V', 'R')
for function_type, merged_df in zip(['T', 'V', 'R'], [train_merged, validation_merged, test_merged]):
    # Create separate figure for each function type
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})  # Increase figure size

    # Set extent to Europe for each subplot
    ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

    # Add natural earth features for coastline and countries for each subplot
    ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

    # Get unique clusters
    clusters = merged_df['Cluster'].unique()

    # Sort clusters
    clusters = sorted(clusters)

    legend_handles = []

    # Plot cities for each cluster
    for j, cluster in enumerate(clusters):
        cluster_data = merged_df[merged_df['Cluster'] == cluster]
        color = cluster_colors[j]
        handle = ax.plot(cluster_data['Longitude'], cluster_data['Latitude'], 'o', color=color, markersize=10, label=legend_labels[j], transform=ccrs.PlateCarree())
        legend_handles.append(handle[0])

    # Add legend with handles for each subplot
    ax.legend(handles=legend_handles, loc='upper right')

    # Add gridlines with labels only on the bottom and left sides for each subplot
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlines = True
    gl.ylines = True
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add title to each subplot
    if function_type == 'T':
        ax.set_title('Train cities: rural T2m difference')
    elif function_type == 'V':
        ax.set_title('Validation cities: rural T2m difference')
    elif function_type == 'R':
        ax.set_title('Test cities: rural T2m difference')

    # Annotate points with values, adjusting for overlap
    texts = []
    for i, row in merged_df.iterrows():
        city = row['city_x']
        value = row['0']
        x, y = row['Longitude'], row['Latitude']

        # Adjust text position for cities close to each other
        close_points = row['close_points']
        if city in ['Rotterdam', 'Utrecht', 'Vienna', 'Gyor']:
            continue

        # Create white box behind text
        rect = patches.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='white', alpha=0.5)
        ax.add_patch(rect)

        # Annotate the point with black text and transparent white background box
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)  # Define transparent textbox properties
        texts.append(ax.text(x, y, f'{value:.2f}', fontsize=10, color='black', fontweight='bold', ha='center', va='bottom', bbox=bbox_props))

    # Adjust text to avoid overlap
    adjust_text(texts, ax=ax)

plt.show()

"""# **Urban diff**"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from adjustText import adjust_text
import matplotlib.patches as patches

# Read the CSV files
validation_path = '/content/drive/MyDrive/thesis/MEANSTAT/urban_difference_per_city_validation.csv'
validation_values = pd.read_csv(validation_path)
train_path = '/content/drive/MyDrive/thesis/MEANSTAT/urban_difference_per_city_train.csv'
train_values = pd.read_csv(train_path)
test_path = '/content/drive/MyDrive/thesis/MEANSTAT/urban_difference_per_city_test.csv'
test_values = pd.read_csv(test_path)

# Read the cluster results CSV file
cluster_path = '/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data.rename(columns={'City': 'city'}, inplace=True)

# Define city groups
city_groups = {'Amsterdam': ['Rotterdam', 'Amsterdam', 'Utrecht'], 'Bratislava': ['Vienna', 'Bratislava', 'Gyor']}

# Calculate mean for Rotterdam group and assign to Rotterdam
Amsterdam_mean = train_values.loc[train_values['city'].isin(['Rotterdam', 'Amsterdam', 'Utrecht']), '0'].mean()
train_values.loc[train_values['city'] == 'Amsterdam', '0'] = Amsterdam_mean

# Calculate mean for Graz group and assign to Graz
Bratislava_mean = train_values.loc[train_values['city'].isin(['Vienna', 'Bratislava', 'Gyor']), '0'].mean()
train_values.loc[train_values['city'] == 'Bratislava', '0'] = Bratislava_mean

# Add a new column containing the first 4 letters of the city name
cluster_data['city_short'] = cluster_data['city'].str[:4]
test_values['city_short'] = test_values['city'].str[:4]
train_values['city_short'] = train_values['city'].str[:4]
validation_values['city_short'] = validation_values['city'].str[:4]

# Merge based on the 'city_short' column
test_merged = pd.merge(cluster_data, test_values, on='city_short')
train_merged = pd.merge(cluster_data, train_values, on='city_short')
validation_merged = pd.merge(cluster_data, validation_values, on='city_short')

# Define colors for clusters
cluster_colors = ['blue', 'green', 'red']

# Define legend labels
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# Function to check if two points are close to each other
def are_close(x1, y1, x2, y2, threshold=1):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold

# Preprocess data to identify clusters of close cities
def preprocess_data(df):
    tree = KDTree(df[['Longitude', 'Latitude']])
    close_points = tree.query_ball_point(df[['Longitude', 'Latitude']], r=1)
    df['close_points'] = close_points

# Preprocess data for each merged dataframe
preprocess_data(train_merged)
preprocess_data(validation_merged)
preprocess_data(test_merged)

# Iterate over each function type ('T', 'V', 'R')
for function_type, merged_df in zip(['T', 'V', 'R'], [train_merged, validation_merged, test_merged]):
    # Create separate figure for each function type
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})  # Increase figure size

    # Set extent to Europe for each subplot
    ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

    # Add natural earth features for coastline and countries for each subplot
    ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

    # Get unique clusters
    clusters = merged_df['Cluster'].unique()

    # Sort clusters
    clusters = sorted(clusters)

    legend_handles = []

    # Plot cities for each cluster
    for j, cluster in enumerate(clusters):
        cluster_data = merged_df[merged_df['Cluster'] == cluster]
        color = cluster_colors[j]
        handle = ax.plot(cluster_data['Longitude'], cluster_data['Latitude'], 'o', color=color, markersize=10, label=legend_labels[j], transform=ccrs.PlateCarree())
        legend_handles.append(handle[0])

    # Add legend with handles for each subplot
    ax.legend(handles=legend_handles, loc='upper right')

    # Add gridlines with labels only on the bottom and left sides for each subplot
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlines = True
    gl.ylines = True
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add title to each subplot
    if function_type == 'T':
        ax.set_title('Train cities: urban T2m difference')
    elif function_type == 'V':
        ax.set_title('Validation cities: urban T2m difference')
    elif function_type == 'R':
        ax.set_title('Test cities: urban T2m difference')

    # Annotate points with values, adjusting for overlap
    texts = []
    for i, row in merged_df.iterrows():
        city = row['city_x']
        value = row['0']
        x, y = row['Longitude'], row['Latitude']

        # Adjust text position for cities close to each other
        close_points = row['close_points']
        if city in ['Rotterdam', 'Utrecht', 'Vienna', 'Gyor']:
            continue

        # Create white box behind text
        rect = patches.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='white', alpha=0.5)
        ax.add_patch(rect)

        # Annotate the point with black text and transparent white background box
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)  # Define transparent textbox properties
        texts.append(ax.text(x, y, f'{value:.2f}', fontsize=10, color='black', fontweight='bold', ha='center', va='bottom', bbox=bbox_props))

    # Adjust text to avoid overlap
    adjust_text(texts, ax=ax)

plt.show()

"""# **urbanminrural diff**"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from adjustText import adjust_text
import matplotlib.patches as patches

# Read the CSV files
validation_path = '/content/drive/MyDrive/thesis/MEANSTAT/urbanminrural_difference_per_city_validation.csv'
validation_values = pd.read_csv(validation_path)
train_path = '/content/drive/MyDrive/thesis/MEANSTAT/urbanminrural_difference_per_city_train.csv'
train_values = pd.read_csv(train_path)
test_path = '/content/drive/MyDrive/thesis/MEANSTAT/urbanminrural_difference_per_city_test.csv'
test_values = pd.read_csv(test_path)

# Read the cluster results CSV file
cluster_path = '/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data.rename(columns={'City': 'city'}, inplace=True)

# Define city groups
city_groups = {'Amsterdam': ['Rotterdam', 'Amsterdam', 'Utrecht'], 'Bratislava': ['Vienna', 'Bratislava', 'Gyor']}

# Calculate mean for Rotterdam group and assign to Rotterdam
Amsterdam_mean = train_values.loc[train_values['city'].isin(['Rotterdam', 'Amsterdam', 'Utrecht']), '0'].mean()
train_values.loc[train_values['city'] == 'Amsterdam', '0'] = Amsterdam_mean

# Calculate mean for Graz group and assign to Graz
Bratislava_mean = train_values.loc[train_values['city'].isin(['Vienna', 'Bratislava', 'Gyor']), '0'].mean()
train_values.loc[train_values['city'] == 'Bratislava', '0'] = Bratislava_mean

# Add a new column containing the first 4 letters of the city name
cluster_data['city_short'] = cluster_data['city'].str[:4]
test_values['city_short'] = test_values['city'].str[:4]
train_values['city_short'] = train_values['city'].str[:4]
validation_values['city_short'] = validation_values['city'].str[:4]

# Merge based on the 'city_short' column
test_merged = pd.merge(cluster_data, test_values, on='city_short')
train_merged = pd.merge(cluster_data, train_values, on='city_short')
validation_merged = pd.merge(cluster_data, validation_values, on='city_short')

# Define colors for clusters
cluster_colors = ['blue', 'green', 'red']

# Define legend labels
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# Function to check if two points are close to each other
def are_close(x1, y1, x2, y2, threshold=1):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold

# Preprocess data to identify clusters of close cities
def preprocess_data(df):
    tree = KDTree(df[['Longitude', 'Latitude']])
    close_points = tree.query_ball_point(df[['Longitude', 'Latitude']], r=1)
    df['close_points'] = close_points

# Preprocess data for each merged dataframe
preprocess_data(train_merged)
preprocess_data(validation_merged)
preprocess_data(test_merged)

# Iterate over each function type ('T', 'V', 'R')
for function_type, merged_df in zip(['T', 'V', 'R'], [train_merged, validation_merged, test_merged]):
    # Create separate figure for each function type
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})  # Increase figure size

    # Set extent to Europe for each subplot
    ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

    # Add natural earth features for coastline and countries for each subplot
    ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

    # Get unique clusters
    clusters = merged_df['Cluster'].unique()

    # Sort clusters
    clusters = sorted(clusters)

    legend_handles = []

    # Plot cities for each cluster
    for j, cluster in enumerate(clusters):
        cluster_data = merged_df[merged_df['Cluster'] == cluster]
        color = cluster_colors[j]
        handle = ax.plot(cluster_data['Longitude'], cluster_data['Latitude'], 'o', color=color, markersize=10, label=legend_labels[j], transform=ccrs.PlateCarree())
        legend_handles.append(handle[0])

    # Add legend with handles for each subplot
    ax.legend(handles=legend_handles, loc='upper right')

    # Add gridlines with labels only on the bottom and left sides for each subplot
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlines = True
    gl.ylines = True
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add title to each subplot
    if function_type == 'T':
        ax.set_title('Train cities: urban-rural T2m ERA5')
    elif function_type == 'V':
        ax.set_title('Validation cities: urban-rural T2m ERA5')
    elif function_type == 'R':
        ax.set_title('Test cities: urban-rural T2m ERA5')

    # Annotate points with values, adjusting for overlap
    texts = []
    for i, row in merged_df.iterrows():
        city = row['city_x']
        value = row['0']
        x, y = row['Longitude'], row['Latitude']

        # Adjust text position for cities close to each other
        close_points = row['close_points']
        if city in ['Rotterdam', 'Utrecht', 'Vienna', 'Gyor']:
            continue

        # Create white box behind text
        rect = patches.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='white', alpha=0.5)
        ax.add_patch(rect)

        # Annotate the point with black text and transparent white background box
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)  # Define transparent textbox properties
        texts.append(ax.text(x, y, f'{value:.2f}', fontsize=10, color='black', fontweight='bold', ha='center', va='bottom', bbox=bbox_props))

    # Adjust text to avoid overlap
    adjust_text(texts, ax=ax)

plt.show()

"""# **STUDY AREA**"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Create a map of Europe in EPSG:4326
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set extent to Europe
ax.set_extent([-23, 40, 35, 70], crs=ccrs.PlateCarree())

# Add natural earth features for coastline and countries
ax.add_feature(NaturalEarthFeature(category='physical', name='coastline', scale='10m', edgecolor='black', facecolor='none'))
ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor='#dbdcdc'))

# Plot cities with white markers and black outline
ax.plot(df['Longitude'], df['Latitude'], 'o', color='white', markersize=6, markeredgecolor='black', transform=ccrs.PlateCarree())

# Add gridlines with labels only on the bottom and left sides
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlines = True
gl.ylines = True
gl.xlabels_bottom = True
gl.ylabels_left = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Set title
#plt.title('Map of Europe with Cities in White')

plt.show()

"""**CREATE THE VALIDATION AND TEST AND TRAIN SETS**"""

import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('/content/drive/MyDrive/thesis/cluster/cluster_results_7MARCH.csv')

# Assuming your DataFrame is named df

# Filter data for train, validation, and test sets based on function
train = df[df['function'] == 'T']
validation = df[df['function'] == 'V']
test = df[df['function'] == 'R']

# Printing length of each set
print("Length of train set:", len(train))
print("Length of validation set:", len(validation))
print("Length of test set:", len(test))

# Saving each set as CSV files
train.to_csv('/content/drive/MyDrive/thesis/cluster/cities_train_7M.csv', index=False)
validation.to_csv('/content/drive/MyDrive/thesis/cluster/cities_validation_7M.csv', index=False)
test.to_csv('/content/drive/MyDrive/thesis/cluster/cities_test_7M.csv', index=False)