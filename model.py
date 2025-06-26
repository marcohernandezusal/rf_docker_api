import sklearn as sk
import pandas as pd
from synth_data import generate_synthetic_data
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Check if there is existing synthetic data
    try:
        synthetic_data = pd.read_csv('synthetic_data.csv')
        print("Synthetic data loaded from 'synthetic_data.csv'.")
    except FileNotFoundError:
        # If not, generate new synthetic data
        print("No existing synthetic data found. Generating new data...")
        synthetic_data = generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1)
        # Save the synthetic data to a CSV file
        synthetic_data.to_csv('synthetic_data.csv', index=False)
        print("Synthetic data generated and saved to 'synthetic_data.csv'.")
    except Exception as e:
        print(f"An error occurred while loading synthetic data: {e}")
        
    # Define a random seed for reproducibility
    random_seed = 42
    
    # Split the data into features and target variable
    X = synthetic_data.drop(columns=['target'])
    y = synthetic_data['target']
    
    # Do a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    
    # Normalize the data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # y_scaler = StandardScaler()
    # y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    # y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Set a Bayesian parameter search for the model
    
    # Define the parameter space for Bayesian optimization
    param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(5, 20),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
        'max_features': Real(0.1, 0.9)
    }
    
    # Initialize the Bayesian search with cross-validation
    bayes_search = BayesSearchCV(
        estimator=RandomForestRegressor(random_state=random_seed),
        search_spaces=param_space,
        n_iter=50,  # Number of iterations for the search
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
        random_state=random_seed,
        n_jobs=-1  # Use all available cores
    )
    
    # Define a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Add a scaler to the pipeline
        ('rf', bayes_search)  # Add the Bayesian search as a step in the pipeline
    ])
    
    # Fit the pipeline to the training data
    print("Starting pipeline fitting...")
    pipeline.fit(X_train, y_train)
    print("Pipeline fitting completed.")
    
    # Print the best parameters found
    print("Best parameters found:")
    print(model.best_params_)
    print(f"Best score (negative MSE): {model.best_score_}")
    # Use the best estimator from the Bayesian search
    model = model.best_estimator_
    
    print("Best estimator model ready for evaluation.")
    
    # Evaluate the model on the test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test R^2: {r2}")
    
    # Save the model using joblib
    import joblib
    joblib.dump(model, 'random_forest_model.pkl')
    print("Model saved to 'random_forest_model.pkl'.")
    
if __name__ == "__main__":
    main()
    