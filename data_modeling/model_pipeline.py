import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DataProcessingMixin:
    """
    Mixin for data processing functionalities including missing data imputation
    and feature addition.
    """
    def fill_missing_data(self):
        """
        Imputes missing data based on the mean for each half-hour of a day.
        Based on unconditional expected value.

        Notes:
           1) Based od previous data exploration, the missing values 
           could not be missing at random. Thus, this naive imputation
           is not a wise choice. But just for simplicity. 

           2) There would also be an issue with day light saving time 
           switch using this approach.
        """
        self.data['half_hour'] = (self.data.index.hour+
                                  self.data.index.minute/30*0.5)
        # Mean for each half hour for every feature
        halfhour_means = self.data.groupby('half_hour').transform('mean')
        # Fill missing values 
        self.data.fillna(halfhour_means, inplace=True)

    def add_features(self, max_lag=48):
        """
        Adds lagged target features to the dataset.

        Args:
            max_lag (int): Maximum number of lags to include.
        """ 
        for lag in range(1, max_lag+1):
            lag_col_name = f'{self.target_variable}_l{lag}'
            self.data[lag_col_name] = self.data[self.target_variable].shift(lag)
        self.data = self.data.dropna()
        

class DataProcessor(DataProcessingMixin):
    """
    Data processor class for preparing time series data.

    Attributes:
        target_variable (str): Target variable in the dataset.
        data (DataFrame): DataFrame containing the time series data.
    """
    def __init__(self, target_variable, data):
        self.target_variable = target_variable
        self.data = data
        
    def process_data(self):
        """
        Processes data by imputing missing values and adding features.

        Returns:
            Tuple[DataFrame, Series]: Processed features and target variable.
        """
        self.fill_missing_data()
        self.add_features()
        # Use only lagged target variables as features
        features = [f for f in self.data.columns
                    if self.target_variable in f and f!=self.target_variable]
        return self.data[features], self.data[self.target_variable]

    @classmethod
    def from_csv(cls, target_variable, filename):
        """
        Creates DataProcessor instance from a CSV file.

        Args:
            target_variable (str): Name of the target variable.
            filename (str): Name of the CSV file.

        Returns:
            DataProcessor: Instance of DataProcessor.
        """
        data = pd.read_csv(
            filename, 
            delimiter=';',
            date_parser=lambda x: pd.to_datetime(x).replace(tzinfo=None),
            index_col='Time').sort_index()
        return cls(target_variable, data)


class ModelEstimator:
    """
    TimeSeries Cross validated Lasso estimator for time series data 
    using Lasso regression with cross-validation.

    Attributes:
        max_lambda (int): Maximum value of lambda for Lasso.
        scaler (StandardScaler): Feature scaler.
        model (Lasso): Trained Lasso model.
    """
    def __init__(self, max_lambda=100):
        """
        Initializes ModelEstimator with specified max_lambda.

        Args:
            max_lambda (int): Maximum lambda for Lasso regularization.
        """
        self.max_lambda = 100
        # For better LASSO convergence
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fits Lasso model to training data using cross-validation.

        Args:
            X (DataFrame): Feature data.
            y (Series): Target variable.

        Returns:
            ModelEstimator: Instance itself.
        """
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            Lasso(), 
            {'alpha': np.linspace(0.01, self.max_lambda, 100)}, 
            cv=tscv, 
            scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)
        self.model = self._fitted_best_model(grid_search.best_params_, X_scaled, y)
        return self

    def predict(self, X):
        """
        Makes predictions using fitted Lasso model.

        Args:
            X (DataFrame): Feature data for predictions.

        Returns:
            ndarray: Predicted values.
        """
        self.predicted = self.model.predict(self.scaler.transform(X))
        return self.predicted

    def _fitted_best_model(self, best_model_params, X, y):
        """
        Fits Lasso model with best parameters from cross-validation.

        Args:
            best_model_params (dict): Best parameters from cross-validation.
            X (DataFrame): Feature data.
            y (Series): Target variable.

        Returns:
            Lasso: Fitted Lasso model.
        """
        return Lasso(**best_model_params).fit(X, y)

    def fitted_model_plot(self, true_y):
        """
        Plots actual vs predicted values.

        Args:
            true_y (pd.Series): True values of the target variable.

        Returns:
            Creates plot within CWD with appropriate name identifier.
        """ 
        result = pd.DataFrame(true_y).rename(columns={true_y.name:'true'})
        result['predicted'] = self.predicted
        plot = result.plot()
        plt.title(f'Predictions of {true_y.name}')
        plt.savefig(f'predictions-{true_y.name}.png')

    def evaluate(self, true_y, naive_predictions):
        """
        Evaluates model performance against naive model.

        Args:
            true_y (pd.Series): True values of target variable.
            naive_predictions (ndarray): Predictions by naive model.

        Prints evaluation metrics and improvements over naive model.
        """
        r_squared = r2_score(true_y, self.predicted)
        improvement_r_squared = r_squared/r2_score(true_y, naive_predictions)-1
        mae = mean_absolute_error(true_y, self.predicted)
        improvement_mae = mean_absolute_error(true_y, naive_predictions)/mae-1
        mse = mean_squared_error(true_y, self.predicted)
        improvement_mse = mean_squared_error(true_y, naive_predictions)/mse-1
        print('Results')
        print('-----------------------------------')
        print('R_squared: {:1.4f}'.format(r_squared))
        print('MAE: {:15.4f}'.format(mae))
        print('MSE: {:19.4f}'.format(mse))
        print('-----------------------------------')
        print('Improvement in R2 over naive model: {:7.4f}'.format(improvement_r_squared))
        print('Improvement in MAE over naive model: {:1.4f}'.format(improvement_mae))
        print('Improvement in MSE over naive model: {:1.4f}'.format(improvement_mse))
