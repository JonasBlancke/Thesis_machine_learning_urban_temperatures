# Thesis: Spatiotemporal Modelling of Urban Air Temperatures over European Cities Using Machine Learning

This repository contains the source code used for my master's thesis in the Master of Science in Bioscience Engineering: Land, Water and Climate. The code is organized into directories according to the structure of the thesis.

## Directory Structure

1. **Preprocessing_features**
    - This directory contains the code for requesting ERA data along with the reprojection and interpolation of the spatial predictor rasters to the target raster of UrbClim.

2. **Subsampling_and_trainvalidationtest_set**
    - The code here merges all the predictors into one dataset (per month and city), applies subsampling, and calculates the solar features.
    - Additionally, scripts are present to convert these datasets per month and city into training, validation, and test sets, and also into a dataset containing all data of all cities and all years.

3. **Cluster_analysis**
    - This script loads the data and runs the K-means clustering process for both the raw data and the PCA-transformed data.

4. **Exploratory_analysis**
    - This directory contains the code to check the performance of ERA5 compared to UrbClim (impact lapse rate correction, general validation indices, per CORINE land cover, etc.).
    - It also includes code to plot this performance per city on a map of Europe.

5. **Hyperparameter_tuning**
    - This directory includes the code to create the subsamples used for hyperparameter tuning, the tuning itself, and scripts for making plots of the results.

6. **Feature_selection**
    - The code here is used to create models for feature selection experiments (prediction files) and to check the validation indices of these models (validation files).
    - Additionally, this directory contains the code for creating feature importance plots and correlation matrices.

7. **General_model_validation**
    - This directory contains the code for the general, spatial, and temporal validation of the model, as well as the code for SHAP violin plots and all other figures present in Section 5.4 of the thesis.

8. **Limited_cities_models**
    - This directory includes the code to create the training, test, and validation datasets for the limited training cities models.
    - Additionally, it contains code for hyperparameter tuning, training, testing, and for generating violin and kernel density/frequency plots for Section 5.5 of the thesis.

## Getting Started

To get started with the code, follow the instructions in the respective directories.

## Contact

For any questions or further information, please contact [Your Name] at [Your Email].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
