![redbul5](https://user-images.githubusercontent.com/32769732/155026150-1ad176c1-4cc2-4999-a1cd-77ac79e28fa3.png)

# #FormulaAI Hack 2022 - Challenge 1: Data Analytics 

Hi, We are Team KuKu and this repository outlines our approach to the [FormulaAI Hackathon](https://github.com/oracle-devrel/formula-ai-2022-hackathon/blob/main/challenges/challenge1.mdd). 

This year's theme was Weather Forecasting for Formula1. The data was provided by the [RedBull Racing eSports Team](https://f1esports.com/pro-championship/teams/red-bull) and It was gathered from the official Formula 1 video game developed by Codemasters. The goal of the task was to find a way to use the deeply nested historical weather data to make accurate weather predictions / forecasts.

## Table of contents
- [FormulaAI Hack 2022 - Challenge 1: Data Analytics]()
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Modelling and Results](#data-modelling-and-results)
- [Future Work](#future-work)

# Getting Started

# Data Preprocessing and Feature Engineering

## Dataset Prepration

The dataset is grouped by the SessionIDs and timestamp. After that it is splitted into train test validation sets. By this way each split has different session_ids. The following features are used for training.

- M_FORECAST_ACCURACY
- M_TOTAL_LAPS
- M_PIT_STOP_WINDOW_IDEAL_LAP
- M_PIT_STOP_WINDOW_LATEST_LAP
- M_PIT_STOP_REJOIN_POSITION
- M_STEERING_ASSIST
- M_BRAKING_ASSIST
- M_GEARBOX_ASSIST
- M_PIT_ASSIST
- M_PIT_RELEASE_ASSIST
- M_ERSASSIST
- M_DRSASSIST
- M_DYNAMIC_RACING_LINE
- M_DYNAMIC_RACING_LINE_TYPE
- M_WEATHER_FORECAST_SAMPLES_M_WEATHER
- M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE
- M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE
- M_TRACK_TEMPERATURE_CHANGE
- M_AIR_TEMPERATURE_CHANGE
- M_RAIN_PERCENTAGE
- Additionally, a column for each flag color is added that shows whether in any of the marshall zones the corresponding flag is up.

To be able to use more training samples we used data points according to time offsets. For example:
- For 5 minutes: we used (0-5), (5-10),(10-15) time offsets.
- For 30 minutes: we used (0-30), (30-60),(60-90), (90,120) time offsets.

One of the key points to mention here is after the feature selection, there are tons of duplicated data entries in the dataset. So, it is very important to delete these duplicated data entries before the train-test splitting operation. Otherwise, there will be lots of exactly same data points in train, validation and test splits which will lead to make evaluations meaningless.

## Feature Engineering

We evaluated importance of features using SHAP library and XGBOOST. We choose the features for training according to following analysis. 

![XGBOOST](/assets/XGBOOST.png)
![SHAP](/assets/SHAP.png)

# Exploratory Data Analysis
![Class Histograms](https://user-images.githubusercontent.com/32769732/155216595-3a8ca014-8e4a-4d89-9f54-405634ba506f.png)

# Data Modelling and Results

## Model

![kuku_diagram](https://user-images.githubusercontent.com/32769732/155241497-39c70c97-cc61-4d51-8f6d-d05c9ad4c8fe.png)



For each time offset we used seperate models for both problems. For rain percentage regression we are feeding the results of the weather predictions as a feature to the input vector. Since the weather predictions are more accurate, and it is very important indicator for rain percentage, it imroved our prediction results for rain percentage prediction.

### Weather

We tried folowing classifiers for weather prediction.

- Logistic Regression
- Decision Tree
- Support Vector Machine
- Random Forest

We chose Random Forest Classifier for weather prediction after comparing the accuracy and f1 scores. 

### Rain Percentage

We tried folowing regressor for rain percentage prediction.

- Linear Regression
- Decision Tree Regressor
- Random Forest regressor
- Multi Layer Perceptron

We chose Random Forest Regressor for rain percentage prediction after comparing the Mean Absolute Error (MAE). 

## Results

### Weather

![](/assets/weather_results.png)
![](/assets/weather_results_classwise.png)


### Rain Percentage
![](/assets/rain.png)

# Future Work

- First of all, after removing the duplicates we have very small amount of data points which leads to poor model performances. 
- Due to small amount of data we could not use the power of Deep Learning algorithms. 
- Due to time limitations we could not try lots of sophisticated models.
