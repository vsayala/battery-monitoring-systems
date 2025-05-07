import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import time
import logging
import uuid  # For generating unique job run IDs

# Configure logging (appends new logs to the same file without deleting old logs)
logging.basicConfig(
    filename="bms-ml-log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'  # Appends logs to the same file
)

# Generate a unique job run ID
job_run_id = str(uuid.uuid4())
logging.info(f"Job Run ID: {job_run_id}")  # Log the job run ID at the start of the execution

# Create a folder for the job run ID to save plots
plots_folder = f"plots_{job_run_id}"
os.makedirs(plots_folder, exist_ok=True)
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
        try:
              logging.info("Performing EDA...")
              numerical_cols = ["Voltage (V)", "Current (A)", "Resistance (Ohms)", "SOC (%)", "SOD (%)", "SOH (%)",
                          "Cell Temperature (°C)", "Ambient Temperature (°C)"]
              logging.info(f"Dataset Info:\n{df.info()}")
              logging.info(f"Summary Statistics:\n{df.describe()}")
              logging.info(f"Missing Values:\n{df.isnull().sum()}")
              logging.info(f"Unique Values in 'State of Cell': {df['State of Cell'].unique()}")
              logging.info(f"Unique Values in 'Communication': {df['Communication'].unique()}")
              logging.info(f"logged Summary Statistics Successfully")     
        except Exception as e:
              logging.error(f"Error during Summary Statistics: {e}")
              raise   
        try:
            logging.info("Plotting histograms and correlation heatmap...")
            try:
                logging.info("Plotting Distribution of Numerical Features...")
                plt.figure(figsize=(15, 10))
                sns.histplot(df[numerical_cols], bins=15, kde=True)
                plt.suptitle("Distribution of Numerical Features")
                dis_path = os.path.join(plots_folder, "distribution_numerical_features.png")
                plt.savefig(dis_path)
                logging.info(f"Saved distribution plot to {dis_path}")
                plt.close()
                logging.info(f"Plotted Summary Statistics Plot Successfully") 
            except Exception as e:
                logging.error(f"Error during Distribution of Numerical Features Plotting: {e}")
                raise
            try:    
                logging.info("Plotting Heatmap...")
                plt.figure(figsize=(10, 8))
                sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Correlation Heatmap")
                heatmap_path = os.path.join(plots_folder, "correlation_heatmap.png")
                plt.savefig(heatmap_path)
                logging.info(f"Saved heatmap to {heatmap_path}")
                plt.close()
                logging.info(f"Plotted Heatmap Successfully")    
            except Exception as e:
                logging.error(f"Error during Heatmap Plotting: {e}")
                raise
            try:
                logging.info("Plotting Pairplot...")
                sns.pairplot(df[numerical_cols])
                plt.suptitle("Pairplot of Numerical Features", y=1.02)
                pair_path = os.path.join(plots_folder, "pairplot_numerical_features.png")
                plt.savefig(pair_path)
                logging.info(f"Saved Pairplot to {pair_path}")
                plt.close()
                logging.info(f"Plotted Pairplot Successfully")
            except Exception as e:
                logging.error(f"Error during Pairplot: {e}")
                raise
            try:
                logging.info("Plotting Boxplot...")
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df[numerical_cols])
                plt.title("Boxplot of Numerical Features")
                box_path = os.path.join(plots_folder, "boxplot_numerical_features.png")
                plt.savefig(box_path)
                logging.info(f"Saved Boxplot to {box_path}")
                plt.close()
                logging.info(f"Plotted Boxplot Successfully")
            except Exception as e:
                logging.error(f"Error during Boxplot: {e}")
                raise
            try:
                logging.info("Plotting Countplot for Communication ...")
                plt.figure(figsize=(10, 6))
                sns.countplot(x="Communication", data=df, palette="viridis")
                plt.title("Distribution of Communication")
                dis_com_path = os.path.join(plots_folder, "countplot_communication.png")
                plt.savefig(dis_com_path)
                logging.info(f"Saved Countplot for Communication to {dis_com_path}")
                plt.close()
                logging.info(f"Plotted Countplot for Communication Successfully")  
            except Exception as e:  
                logging.error(f"Error during Countplot: {e}")
                raise
            try:    
                logging.info("Plotting Countplot for State of Cell...")
                plt.figure(figsize=(10, 6))
                sns.countplot(x="State of Cell", data=df, palette="viridis")
                plt.title("Distribution of State of Cell")
                dis_state_path = os.path.join(plots_folder, "countplot_state_of_cell.png")
                plt.savefig(dis_state_path)
                logging.info(f"Saved Countplot for State of Cell to {dis_state_path}")
                plt.close()
                logging.info(f"Plotted Countplot for State of Cell Successfully")  
            except Exception as e:  
                logging.error(f"Error during Countplot: {e}")
                raise
        except Exception as e:
            logging.error(f"Error during plotting: {e}")
            raise
        try:
            # Trend analysis for a sample Cell ID
            logging.info("Performing trend analysis for a sample Cell ID...")
            sample_cell_id = df["Cell ID"].iloc[0]
            sample_data = df[df["Cell ID"] == sample_cell_id].sort_values("Timestamp")
            plt.figure(figsize=(12, 6))
            plt.plot(sample_data["Timestamp"], sample_data["Voltage (V)"], label="Voltage (V)")
            plt.plot(sample_data["Timestamp"], sample_data["Current (A)"], label="Current (A)")
            plt.plot(sample_data["Timestamp"], sample_data["SOH (%)"], label="SOH (%)")
            plt.title(f"Trends for Cell ID: {sample_cell_id}")
            plt.xlabel("Timestamp")
            plt.ylabel("Values")
            plt.legend()
            plt.grid(True)
            sample_cell_path = os.path.join(plots_folder, "trend_analysis_sample_cell.png")
            plt.savefig(sample_cell_path)
            logging.info(f"Saved Trend Analysis of Sample Cell to {sample_cell_path}")
            plt.close()
            logging.info(f"Plotted Trend Analysis of Sample Cell Successfully")
        except Exception as e:
            logging.error(f"Error during trend analysis: {e}")
            raise
        logging.info("EDA completed successfully.")
    except Exception as e:
        logging.error(f"Error during EDA: {e}")
        raise

@timing_decorator
def preprocess_data(df):
    """Preprocess data by imputing missing values and scaling features."""
    try:
        numerical_cols = ["Voltage (V)", "Current (A)", "Resistance (Ohms)", "SOC (%)", "SOD (%)", "SOH (%)",
                          "Cell Temperature (°C)", "Ambient Temperature (°C)"]
        imputer = SimpleImputer(strategy="mean")

        scalers = {}

        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        # Scale each column individually and store the scaler
        for col in numerical_cols:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler  # Store the scaler for reversing later

        logging.info("Data preprocessing completed successfully.")
        return df, scalers
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
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
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        }

        logging.info(f"Starting hyperparameter tuning for {target_name} with the following param_grid:")
        for param, values in param_grid.items():
            logging.info(f"    {param}: {values}")

        gbr = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring="r2", cv=3, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        logging.info(f"Hyperparameter tuning for {target_name} completed successfully.")
        logging.info(f"Best hyperparameters for {target_name}: {grid_search.best_params_}")
        return grid_search.best_params_
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning for {target_name}: {e}")
        raise

@timing_decorator
def train_time_series_models(df, target_columns):
    """Train time-series models for predicting the next values of the targets."""
    try:
        results = {}
        feature_columns = [col for col in df.columns if "lag" in col]  # Use lag features for prediction

        for target in target_columns:
            logging.info(f"Training time-series model for {target}...")
            X = df[feature_columns]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Tune hyperparameters for the target variable
            best_params = tune_hyperparameters(X_train, y_train, target)

            # Train the regression model with the best parameters
            best_model = GradientBoostingRegressor(random_state=42, **best_params)
            best_model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Evaluation metrics for {target}: MSE={mse:.4f}, R2={r2:.4f}")
            print(f"{target} - Mean Squared Error: {mse:.4f}, R2 Score: {r2:.4f}")

            # Save the trained model and evaluation results
            results[target] = {"model": best_model, "mse": mse, "r2": r2}

        logging.info("All time-series models trained successfully.")
        return results
    except Exception as e:
        logging.error(f"Error during time-series model training: {e}")
        raise

@timing_decorator
def predict_next_values(models, df, feature_columns,scalers, target_columns):
    """Predict the next values for all target columns."""
    try:
        predictions = {}
        latest_features = df.iloc[-1][feature_columns].values.reshape(1, -1)  # Get the latest row of features
        for target, data in models.items():
            model = data["model"]
            prediction = model.predict(latest_features)[0]
            # Reverse scaling for the predicted value
            original_value = reverse_scaling(scalers[target], np.array([prediction]), target)
            predictions[target] = original_value[0]  # Store the original value
            logging.info(f"Predicted next value for {target}: {prediction:.4f}")
        return predictions
    except Exception as e:
        logging.error(f"Error during next value prediction: {e}")
        raise

@timing_decorator
def plot_predictions(df, predictions, target_columns, scalers, cell_id):
    """
    Plot the historical and predicted values for each target column in their original scale.

    Args:
        df (pd.DataFrame): DataFrame containing the historical data.
        predictions (dict): Dictionary of predicted values for each target column.
        target_columns (list): List of target columns to plot.
        scalers (dict): Dictionary of MinMaxScaler objects for each target column.
        cell_id (str): Cell ID under consideration for plotting.
    """
    try:
        # Filter the DataFrame for the specific cell
        cell_data = df[df["Cell ID"] == cell_id]

        # Create subplots in a 3x2 layout
        num_targets = len(target_columns)
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        axes = axes.flatten()  # Flatten the 3x2 grid for easier indexing

        for i, target in enumerate(target_columns):
            ax = axes[i]

            # Reverse scaling for historical values
            scaler = scalers[target]
            historical_values = scaler.inverse_transform(cell_data[[target]]).flatten()

            # Plot historical values
            ax.plot(
                cell_data["Timestamp"],
                historical_values,
                label=f"Historical {target}", color="blue", alpha=0.7
            )

            # Plot the predicted value
            ax.scatter(
                [cell_data["Timestamp"].iloc[-1]],  # Use the last timestamp for the predicted value
                [predictions[target]],  # Use the predicted value
                label=f"Predicted {target}", color="red", s=100, zorder=5
            )

            # Set plot labels and title
            ax.set_title(f"{target} Over Time for Cell {cell_id}", fontsize=14)
            ax.set_xlabel("Timestamp", fontsize=12)
            ax.set_ylabel(target, fontsize=12)
            ax.legend()
            ax.grid(True)

        # Adjust layout for better appearance
        plt.tight_layout()

        # Save the plot to the plots folder
        plot_path = os.path.join(plots_folder, f"predicted_values_plot_cell_{cell_id}.png")
        plt.savefig(plot_path)
        logging.info(f"Saved predicted values plot to {plot_path}")

        # Show the plot
        plt.show()

    except Exception as e:
        logging.error(f"Error during plotting predictions: {e}")
        raise

# Main Workflow
if __name__ == "__main__":
    try:
        overall_start = time.time()
        file_path = "dummy_battery_data.csv"
        df = load_data(file_path)
        perform_eda(df)
        # Preprocess data
        df, scalers = preprocess_data(df)

        # Define target columns
        target_columns = ["Voltage (V)", "Current (A)", "Resistance (Ohms)", "SOC (%)", "SOD (%)", "SOH (%)",
                          "Cell Temperature (°C)", "Ambient Temperature (°C)"]

        # Create lag features for time-series prediction
        df = create_lag_features(df, target_columns, lags=3)

        # Train time-series models
        models = train_time_series_models(df, target_columns)

        # Predict the next values for cell 1234
        feature_columns = [col for col in df.columns if "lag" in col]
        predictions = predict_next_values(models, df, feature_columns, scalers, target_columns)

        print("\nPredicted Next Values for Cell 1234:")
        for target, value in predictions.items():
            print(f"  {target}: {value:.4f}")
        logging.info(f"Predicted Next Values for Cell 1234: {predictions}")
        # Plot historical and predicted values
        cell_id = "CELL1234"  # Example cell ID
        # Ensure the cell_id exists in the DataFrame
        if cell_id not in df["Cell ID"].values:
            logging.error(f"Cell ID {cell_id} not found in the DataFrame.")
            raise ValueError(f"Cell ID {cell_id} not found in the DataFrame.")
        else:
            logging.info(f"Cell ID {cell_id} found in the DataFrame.")
        # Plot predictions for the specified cell ID
        plot_predictions(df, predictions, target_columns,scalers,cell_id)
        logging.info(f"Total execution time: {time.time() - overall_start:.2f} seconds")
    except Exception as e:
        logging.critical(f"Critical error: {e}")