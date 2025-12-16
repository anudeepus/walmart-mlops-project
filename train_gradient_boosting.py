import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup MLflow
mlflow.set_tracking_uri("http://0.0.0.0:5000") 
mlflow.set_experiment("Walmart_Sales_Prediction")

def train_and_log():
    print("Loading data...")
    train = pd.read_csv("train.csv")
    validate = pd.read_csv("validate.csv")

    # Preprocess
    target = "Weekly_Sales"
    drop_cols = ["Date", target]
    
    X_train = train.drop(columns=drop_cols)
    y_train = train[target]
    
    X_val = validate.drop(columns=drop_cols)
    y_val = validate[target]

    # Start Run
    with mlflow.start_run(run_name="Gradient_Boosting_Model"):
        print("Training Gradient Boosting...")
        
        # Define Model
        # n_estimators=100, learning_rate=0.1 are standard starting points
        n_estimators = 100
        learning_rate = 0.1
        max_depth = 3
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            random_state=42
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_val)
        
        # Metrics
        mse = mean_squared_error(y_val, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

        # Log to MLflow
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")
        print("Gradient Boosting Model logged to MLflow!")

if __name__ == "__main__":
    train_and_log()
