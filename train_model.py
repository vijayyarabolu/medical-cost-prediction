import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import feature_extraction

def train_and_log_model(data_path: str, model_type: str = "linear"):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Apply feature engineering
    df = feature_extraction.build_feature_pipeline(df)
    
    # Prepare features and target
    # Convert categorical to numeric for simple models
    df_processed = pd.get_dummies(df, columns=["sex", "smoker", "region", "bmi_category", "age_group"], drop_first=True)
    
    X = df_processed.drop(["charges", "text"], axis=1, errors="ignore") # Drop text if present
    y = df_processed["charges"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    mlflow.set_experiment("Medical_Cost_Prediction")
    
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_path", data_path)
        
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            n_estimators = 100
            max_depth = 10
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            raise ValueError("Unknown model type")
            
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model: {model_type}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")

if __name__ == "__main__":
    # Ensure data exists (using the cleaned one or original)
    # Assuming 'cleaned_insurance_data.csv' exists from previous steps or we use 'insurance.csv'
    # For this script, let's use 'insurance.csv' and let the pipeline handle it, 
    # but the feature_extraction expects certain columns.
    
    # Let's try to use 'insurance.csv' directly as it's the source
    train_and_log_model("insurance.csv", "linear")
    train_and_log_model("insurance.csv", "random_forest")
