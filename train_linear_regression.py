import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup MLflow
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("Walmart_Sales_Prediction")

def train_lr():
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

    with mlflow.start_run(run_name="Linear_Regression_Model"):
        print("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_val)
        
        # Metrics
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")

        # Log
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
        print("Linear Regression Logged!")

if __name__ == "__main__":
    train_lr()

