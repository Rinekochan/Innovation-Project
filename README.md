# AI-based Project - Assignment 2

This project focuses on training a machine learning model to predict flight prices, clustering similar flights for recommendation feature, and classify the flight price based on historical datas. Below are the instructions to set up the project environment, process the data, train the model, and use it for prediction.

## 1. Project Setup

### 1.1. Prerequisites

Ensure you have Python 3.8+ installed on your machine. You can check your Python version by running:

```bash
python --version
```

If not, you can download lastest Python in https://www.python.org/downloads/

### 1.2. VSCode Setup

Ensure you have [VSCode](https://code.visualstudio.com/) installed on your machine, then you can download these extensions: Python, Pylance, Jupyter

### 1.3. Dependencies/Packages Setup

If your Python installation doesn't include pip, you can install it [here](https://packaging.python.org/en/latest/tutorials/installing-packages/) 

You need to install these packages using **pip** in the project in order to run our Jupyter notebook:

```bash
pip install pandas
pip install matplotlib
pip install seaborn
pip install numpy
pip install scikit-learn
pip install lightgbm
```

If you have errors with pip install on Windows, please check if system environment path contains the path to Python yet. 

## 2. Model Training Instructions
### 2.1. Datasets
- There are 5 datasets being used in this project, storing in **/datasets** folder:
    - **US_Airfare.csv**: Main dataset of the project, contains USA flights from 1997 to 2024
    - **US_AirTraffic.csv**: Contains total domestic passengers and flights, showing the trend of seasonal commercial flights.
    - **US_GDP.csv**: Contains the GDP of USA from 1997 to 2023.
    - **US_Inflation.csv**: Contains the Consumer Price Index (CPI) of USA from 1947 to 2023
    - **US_JetFuelDomestic.xlsx**: Contains the fuel price in USA from 2000 to 2024
### 2.2. Data processing
- In the **/notebooks** folder, our team have created a Jupyter notebook. You'll find detailed explaination of how we process data from 5 datasets.
    - Open models.ipynb
    - The notebook guides you through data preprocessing such as handling missing values, feature engineering, and normalization.
- In summary, we have combined all datasets into a dataset based on Year and Quarter since 2014. Then we handle missing values either by filling lastest values or calculating linear trending values.
### 2.3. Data visualisation
- In the same notebook, we have also visualised and demonstrated relationships between fields in the dataset. Then we also use RandomForestRegressor for feature engineering by showing what field is important to the model.
### 2.4. Model training 
- Training the model is also done within the same notebook. The process is detailed, and all code is included in the cells. We conduct three different models:
    - **Random Forest** for regression, predicting flight prices
    - **DBScan** for clustering, grouping similar flights for flights suggestion system
    - **LightGBM** for classification, classify the pricing categories of the flight in that specific route.
### 2.5. Model Evaluation and Prediction
- Once the model is trained, evaluate its performance and use it to make predictions on new data — all in the same notebook.
- The notebook includes cells to measure performance metrics like accuracy, precision, and recall after training each model.

## How to run this notebook?
There are two ways to run this notebook:
- Click Run All on the top of the Jupyter file in VSCode.
- Click on Execute Cell button next to every cells.

## Reference
- [scikit-learn Documentation](https://scikit-learn.org/stable/install.html)
- [pandas Documentation](https://pandas.pydata.org/docs/getting_started/index.html)
- [Jupyter Documentation](https://jupyter.org/)
