import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from fancyimpute import IterativeImputer  # For MICE imputation
import time
import logging
import uuid  # For generating unique job run IDs
import sys
from xgboost import XGBRegressor

# Configure logging (appends new logs to the same file without deleting old logs)
logging.basicConfig(
    filename="../../logs_bms/bms_ml_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'  # Appends logs to the same file
)

# Generate a unique job run ID including the date and timestamp
current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
unique_id = str(uuid.uuid4())
job_run_id = f"{current_datetime}_{unique_id}"

# Extract year, month, and day from the current date
current_year = time.strftime("%Y")
current_month = time.strftime("%m")
current_day = time.strftime("%d")

# Create a folder structure: year/month/day/job_run_id
# This will create a nested folder structure for organizing plots

plots_folder = os.path.join(current_year, current_month, current_day, job_run_id)
os.makedirs(plots_folder, exist_ok=True)
logging.info(f"Created folder structure for job run ID: {job_run_id}")

# Log the folder creation
logging.info(f"Job Run ID: {job_run_id}")  # Log the job run ID at the start of the execution
logging.info(f"Created folder for plots: {plots_folder}")

def timing_decorator(func):
    """Decorator to measure execution time of a function and log it."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Started execution of '{func.__name__}'")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in '{func.__name__}': {e}")
            raise e
        else:
            elapsed_time = time.time() - start_time
            logging.info(f"Completed '{func.__name__}' in {elapsed_time:.2f} seconds")
            return result
    return wrapper

@timing_decorator
def load_data(file_path):
    """Load the dataset and parse the Timestamp column."""
    try:
        df = pd.read_csv(file_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during data loading: {e}")
        raise

@timing_decorator
def perform_eda(df):
    """Perform Exploratory Data Analysis (EDA)."""
    try:
        logging.info("Performing EDA...")
        numerical_cols = ["Voltage (V)", "Current (A)", "Resistance (Ohms)", "SOC (%)", "SOD (%)", "SOH (%)",
                          "Cell Temperature (°C)", "Ambient Temperature (°C)"]
        logging.info(f"Dataset Info:\n{df.info()}")
        logging.info(f"Summary Statistics:\n{df.describe()}")
        logging.info(f"Missing Values:\n{df.isnull().sum()}")
        logging.info(f"Unique Values in 'State of Cell': {df['State of Cell'].unique()}")
        logging.info(f"Unique Values in 'Communication': {df['Communication'].unique()}")

        # Plot data distributions and correlations
        plt.figure(figsize=(15, 10))
        sns.histplot(df[numerical_cols], bins=15, kde=True)
        plt.suptitle("Distribution of Numerical Features")
        dis_path = os.path.join(plots_folder, "distribution_numerical_features.png")
        plt.savefig(dis_path)
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(plots_folder, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()

        logging.info(f"EDA plots saved to {plots_folder}")
    except Exception as e:
        logging.error(f"Error during EDA: {e}")
        raise

@timing_decorator
def preprocess_missing_data(df, target_columns):
    """
    Preprocess missing data by filling gaps caused by communication="no".
    Temporarily impute missing values for prediction using interpolation.
    """
    try:
        logging.info("Handling missing data caused by communication='no'.")
        for col in target_columns:
            if col in df.columns:
                # Use interpolation to fill gaps (linear method)
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                # If interpolation fails for any gaps, use forward and backward fill
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
        logging.info("Missing data handled successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during missing data handling: {e}")
        raise

@timing_decorator
def create_lag_features(df, target_columns, lags=1):
    """Create lag features for time-series forecasting."""
    try:
        df = df.copy()
        for target in target_columns:
            for lag in range(1, lags + 1):
                df[f"{target}_lag{lag}"] = df[target].shift(lag)
        df = df.dropna()  # Drop rows with NaN values introduced by lags
        logging.info(f"Lag features created successfully for targets: {target_columns}")
        return df
    except Exception as e:
        logging.error(f"Error during lag feature creation: {e}")
        raise

@timing_decorator
def tune_hyperparameters(X_train, y_train, target_name):
    """Tune hyperparameters for regression models and log each param grid."""
    try:
        param_grid = {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.001, 0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 6],
            "subsample": [0.6, 0.8, 1.0],
            "max_features": [None, "sqrt", "log2"],
        }

        logging.info(f"Starting hyperparameter tuning for {target_name} with the following param_grid:")
        for param, values in param_grid.items():
            logging.info(f"    {param}: {values}")

        # Dynamically determine the number of splits for cross-validation
        n_splits = min(5, len(X_train))
        if n_splits < 2:
            raise ValueError(f"Not enough samples to perform cross-validation for '{target_name}' (n_samples={len(X_train)}).")

        gbr = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring="r2", cv=n_splits, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        logging.info(f"Hyperparameter tuning for {target_name} completed successfully.")
        logging.info(f"Best hyperparameters for {target_name}: {grid_search.best_params_}")
        return grid_search.best_params_
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning for {target_name}: {e}")
        raise

@timing_decorator
def reverse_scaling(scaler, predictions, column_name):
    """Reverse the MinMaxScaler transformation."""
    try:
        # Convert to 2D array and reverse the scaling
        scaled_array = predictions.reshape(-1, 1)
        return scaler.inverse_transform(scaled_array).flatten()  # Flatten back to 1D
    except Exception as e:
        logging.error(f"Error during reverse scaling for {column_name}: {e}")
        raise

@timing_decorator
def train_advanced_time_series_models(df, target_columns):
    """
    Train advanced time-series models (e.g., XGBoost) for forecasting.
    """
    try:
        results = {}
        feature_columns = [col for col in df.columns if "lag" in col]

        for target in target_columns:
            logging.info(f"Training advanced time-series model for {target}...")
            X = df[feature_columns]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Use XGBoost for time-series forecasting
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Evaluation metrics for {target}: MSE={mse:.4f}, R2={r2:.4f}")
            print(f"{target} - Mean Squared Error: {mse:.4f}, R2 Score: {r2:.4f}")

            # Save the trained model and evaluation results
            results[target] = {"model": model, "mse": mse, "r2": r2}

        logging.info("All advanced time-series models trained successfully.")
        return results
    except Exception as e:
        logging.error(f"Error during advanced time-series model training: {e}")
        raise

@timing_decorator
def predict_next_values_with_imputation(models, df, feature_columns, scalers, target_columns, steps=10):
    """
    Predict the next values for all target columns, using imputed values if necessary.
    """
    try:
        predictions = {}
        latest_features = df.iloc[-1][feature_columns].values.reshape(1, -1)
        for target, data in models.items():
            model = data["model"]
            prediction = []

            for _ in range(steps):
                next_prediction = model.predict(latest_features)[0]
                prediction.append(next_prediction)

                # Update features with the predicted value for the next step
                latest_features = np.append(latest_features[:, 1:], [[next_prediction]], axis=1)

            predictions[target] = prediction
            logging.info(f"Predicted next values for {target}: {prediction}")

        return predictions
    except Exception as e:
        logging.error(f"Error during next value prediction with imputation: {e}")
        raise

@timing_decorator
def train_failure_prediction_model(df, target_column):
    """Train a model to predict cell failures."""
    try:
        # Prepare the data
        X = df.drop(columns=[target_column])
        y = df[target_column]  # Binary labels: 0 (Healthy), 1 (Dead)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a classification model
        model = XGBClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        logging.info(f"Failure Prediction Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        return model
    except Exception as e:
        logging.error(f"Error during failure prediction model training: {e}")
        raise

@timing_decorator
def predict_failure_cells(model, df):
    """Predict cells likely to fail."""
    try:
        predictions = model.predict(df)
        failure_cells = df.loc[predictions == 1, "Cell ID"].unique()
        logging.info(f"Cells predicted to fail: {failure_cells}")
        return failure_cells
    except Exception as e:
        logging.error(f"Error predicting cell failures: {e}")
        raise

@timing_decorator
def plot_predictions(df, predictions, target_columns, scalers, cell_id, steps=10):
    """
    Plot the historical and predicted values for each target column in their original scale.
    """
    try:
        cell_data = df[df["Cell ID"] == cell_id]
        num_targets = len(target_columns)
        n_cols = 2
        n_rows = (num_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
        axes = axes.flatten()

        for i, target in enumerate(target_columns):
            ax = axes[i]
            ax.plot(cell_data["Timestamp"], cell_data[target], label=f"Historical {target}", color="blue", alpha=0.7)
            ax.scatter(
                pd.date_range(start=cell_data["Timestamp"].iloc[-1], periods=steps, freq="T"),
                predictions[target],
                label=f"Predicted {target}", color="red", s=100
            )
            ax.set_title(f"{target} Over Time for Cell {cell_id}")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(plots_folder, f"predicted_values_plot_cell_{cell_id}.png")
        plt.savefig(plot_path)
        plt.show()
    except Exception as e:
        logging.error(f"Error during plotting predictions: {e}")
        raise

# Main Workflow
if __name__ == "__main__":
    try:
        overall_start = time.time()

        if len(sys.argv) < 2:
            raise ValueError("Please provide a valid cell_id as an input argument.")
        cell_id = sys.argv[1]

        file_path = "dummy_battery_data.csv"
        df = load_data(file_path)

        if cell_id not in df["Cell ID"].values:
            logging.error(f"Cell ID {cell_id} not found in the DataFrame.")
            raise ValueError(f"Cell ID {cell_id} not found in the DataFrame.")

        perform_eda(df)
        df = df[df["Cell ID"] == cell_id]

        target_columns = ["Voltage (V)", "Current (A)", "Resistance (Ohms)", "SOC (%)", "SOD (%)", "SOH (%)",
                          "Cell Temperature (°C)", "Ambient Temperature (°C)"]
        df = preprocess_missing_data(df, target_columns)
        df = create_lag_features(df, target_columns, lags=3)

        models = train_advanced_time_series_models(df, target_columns)
        feature_columns = [col for col in df.columns if "lag" in col]
        predictions = predict_next_values_with_imputation(models, df, feature_columns, {}, target_columns)

        print(f"\nPredicted Next Values for Cell {cell_id}:")
        for target, values in predictions.items():
            print(f"  {target}: {values}")
        plot_predictions(df, predictions, target_columns, {}, cell_id)

        # Train failure prediction model
        failure_model = train_failure_prediction_model(df, target_column="IsDead")  # Add an "IsDead" column to your dataset

        # Predict failing cells
        failing_cells = predict_failure_cells(failure_model, df)
        
        print(f"Cells likely to fail: {failing_cells}")
        logging.info(f"Total execution time: {time.time() - overall_start:.2f} seconds")
    except Exception as e:
        logging.critical(f"Critical error: {e}")