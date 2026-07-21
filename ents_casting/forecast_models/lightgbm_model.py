"""Contains functions to tune, train, save and predict with LightGBM models"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import lightgbm as lgb
import optuna
from abc import ABC, abstractmethod
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

from ents_casting.config import config


# Abstract base class for Lgbm prediction models
class LgbmForecastModel(ABC):
    ts_param: str
    input_parameters: List[str]
    model: lgb.Booster

    @abstractmethod
    def __init__(self, ts_param: str) -> None:
        # Needs to be implemented in child classes
        pass

    @abstractmethod
    def load_data(self, year: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Loads the data for training or testing the model.

        Args:
            year (int): Year of the measured data to load.
        Returns:
            tuple[np.ndarray, dict[str, np.ndarray]]: Target values (Y_data) and input features (X_data).
        """
        # Needs to be implemented in child classes
        pass

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """
        Run the LightGBM model to predict the time series parameter based on the input data.

        Args:
            data (Dict[str, np.ndarray]): Dictionary of 1-D arrays

        Returns:
            np.ndarray: 1-D array of time series parameter values for each time step.
        """
        # Check if required input parameters are present
        for param in self.input_parameters:
            if param not in data:
                raise ValueError(f"Missing required input parameter: {param}")

        # Prepare input data for prediction
        input_data_np = np.zeros((len(data[self.input_parameters[0]]), len(self.input_parameters)))
        for i, param in enumerate(self.input_parameters):
            input_data_np[:, i] = data[param]

        # Make predictions
        predictions = self.model.predict(input_data_np)
        return predictions

    def predict_with_lag(self, data: dict[str, np.ndarray], lag_values: List[float]) -> np.ndarray:
        """
        Run the LightGBM model to predict the time series parameter based on the input data. Lagged
        target variable values are included in the input features and updated after each prediction.

        Args:
            data (Dict[str, np.ndarray]): Dictionary of 1-D arrays
            lag_values (List[float]): 1-D array of lagged target variable values, ending with the most recent value.

        Returns:
            np.ndarray: 1-D array of time series parameter values for each time step.
        """
        # Check if required input parameters are present
        for param in self.input_parameters:
            if param not in data:
                if param.startswith("lag_"):
                    # Create empty array for lag features if not present
                    data[param] = np.zeros(len(data[self.input_parameters[0]]))
                else:
                    raise ValueError(f"Missing required input parameter: {param}")

        time_steps = len(data[self.input_parameters[0]])
        predictions = np.zeros(time_steps)

        lag_feature_funcs = {
            "lag_mean_24": lambda vals: float(np.mean(vals[-24:])),
            "lag_min_24": lambda vals: float(np.min(vals[-24:])),
            "lag_max_24": lambda vals: float(np.max(vals[-24:])),
            "lag_mean_168": lambda vals: float(np.mean(vals[-168:])),
            "lag_min_168": lambda vals: float(np.min(vals[-168:])),
            "lag_max_168": lambda vals: float(np.max(vals[-168:])),
        }
        lag_feature_indices = [i for i, p in enumerate(self.input_parameters) if p in lag_feature_funcs]
        non_lag_indices = [i for i, p in enumerate(self.input_parameters) if p not in lag_feature_funcs]

        for t in range(time_steps):
            input_data_np = np.zeros((1, len(self.input_parameters)))

            # Fill non-lag features directly from data
            for i in non_lag_indices:
                param = self.input_parameters[i]
                input_data_np[0, i] = data[param][t]

            # Compute lag features only for lag indices
            for i in lag_feature_indices:
                param = self.input_parameters[i]
                input_data_np[0, i] = lag_feature_funcs[param](lag_values)

            # Make predictions
            predictions[t] = self.model.predict(input_data_np)[0]
            lag_values.append(predictions[t])

        return predictions

    def train(
        self,
        X_data: np.ndarray,
        Y_data: np.ndarray,
        params: dict[str, Any],
    ) -> lgb.Booster:
        """
        Train LightBGM model with all available data and the best hyperparameters.

        Args:
            X_data (np.ndarray): 2D array of input features.
            Y_data (np.ndarray): 1D array of target variable.
            params (dict[str, Any]): Hyperparameters for LightGBM.
        """
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "learning_rate": params["learning_rate"],
            "num_leaves": params["num_leaves"],
            "max_depth": params["max_depth"],
            "num_boost_round": params["num_boost_round"],
        }
        dtrain = lgb.Dataset(X_data, label=Y_data)
        self.model = lgb.train(params, dtrain)

    def load_gbm_model(self, fold: int = 0) -> lgb.Booster:
        """Load a pre-trained LightGBM model."""
        model_path = Path.cwd() / config["path_data_forecast_model_params"] / f"{self.ts_param}_model_fold_{fold}.txt"
        if not model_path.exists():
            model_path = Path.cwd() / config["path_data_forecast_model_params"] / f"{self.ts_param}_model.txt"

        self.model = lgb.Booster(model_file=model_path)
        return self.model

    def tune_hyperparameters(
        self,
        year: int,
        n_trials: int = 30,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna with k-fold cross-validation.

        Args:
            year (int): Year of the measured data to use for validation.
            n_trials (int): Number of Optuna trials.
            seed (int): Random seed for reproducibility.

        Returns:
            dict[str, Any]: Best hyperparameters found by Optuna.
        """
        Y_data, X_data = self.load_data(year)

        # Cutoff the first 168 hours due to lag features
        Y_data = Y_data[168:8760]
        for key in X_data:
            X_data[key] = X_data[key][168:8760]

        # Convert to numpy arrays for CV
        y = np.asarray(Y_data)
        X = np.asarray(list(X_data.values())).T

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)

        def objective(trial) -> float:
            """Objective function for Optuna hyperparameter optimization."""
            params: Dict[str, Any] = {
                "objective": "regression",
                "metric": "l2",
                "verbosity": -1,
                "seed": seed,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
                "max_depth": trial.suggest_int("max_depth", 6, 16),
            }
            # Store num_boost_round as a tunable hyperparameter as well
            num_boost_round = trial.suggest_int("num_boost_round", 100, 2000, log=True)

            rmses: list[float] = []

            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[valid_data],
                )

                y_pred = model.predict(X_valid)
                rmse = root_mean_squared_error(y_valid, y_pred)
                rmses.append(rmse)

            # Return mean RMSE across folds
            return float(np.mean(rmses))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Print results of the best trial
        best_trial = study.best_trial
        print(f"Best trial mean k-fold RMSE for {self.ts_param}: {best_trial.value:.4f}")
        print(f"Best hyperparameters for {self.ts_param}:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        # Save best hyperparameters to a JSON file
        with open(
            Path.cwd() / config["path_data_forecast_model_params"] / f"{self.ts_param}_model_best_params.json",
            "w",
        ) as f:
            json.dump(best_trial.params, f, indent=2)

        return best_trial.params

    def cross_validate_and_save_models(self, year: int):
        """
        Train a LightGBM model for each fold, save each fold model, and print cross-validation results.

        Args:
            year (int): Year of the measured data to load.

        """
        Y_data, X_data = self.load_data(year)

        # Cutoff the first 168 hours due to lag features
        Y_data = Y_data[168:8760]
        for key in X_data:
            X_data[key] = X_data[key][168:8760]

        # Convert to numpy arrays for training and validation
        y = np.asarray(Y_data)
        X = np.asarray(list(X_data.values())).T

        with open(
            Path.cwd() / config["path_data_forecast_model_params"] / f"{self.ts_param}_model_best_params.json",
            "r",
        ) as f:
            best_params = json.load(f)

        n_days_per_fold = config["n_days_per_cross_validation_fold"]
        fold_length = n_days_per_fold * 24
        model_dir = Path.cwd() / config["path_data_forecast_model_params"]

        r2_scores: list[float] = []
        mae_scores: list[float] = []
        rmse_scores: list[float] = []

        for fold_idx, prediction_interval_start in enumerate(range(0, len(y), fold_length)):
            prediction_interval_end = min(prediction_interval_start + fold_length, len(y))

            X_train = np.concatenate((X[:prediction_interval_start], X[prediction_interval_end:]), axis=0)
            Y_train = np.concatenate((y[:prediction_interval_start], y[prediction_interval_end:]), axis=0)
            X_valid = X[prediction_interval_start:prediction_interval_end]
            y_valid = y[prediction_interval_start:prediction_interval_end]

            self.train(X_train, Y_train, best_params)

            file = model_dir / f"{self.ts_param}_model_fold_{fold_idx}.txt"
            self.model.save_model(file)
            print(f"Saved trained model for {self.ts_param} fold {fold_idx} in {file}.")

            y_pred = self.model.predict(X_valid)
            r2_scores.append(r2_score(y_valid, y_pred))
            mae_scores.append(mean_absolute_error(y_valid, y_pred))
            rmse_scores.append(root_mean_squared_error(y_valid, y_pred))

        print(f"Cross-validation results for parameter {self.ts_param} (perfect weather forecasts):")
        print(f"Average R2 Score: {np.mean(r2_scores):.4f}")
        print(f"Average MAE: {np.mean(mae_scores):.4f}")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
