from sklearn.datasets import make_regression
import pandas as pd

def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1, targets=True):
    """
    Generate synthetic regression data.

    Parameters:
    n_samples (int): Number of samples to generate.
    n_features (int): Number of features to generate.
    noise (float): Standard deviation of the Gaussian noise added to the output.

    Returns:
    pd.DataFrame: DataFrame containing the features and target variable.
    """
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    if not targets:
        df = df.drop(columns=['target'])
    
    return df

if __name__ == "__main__":
    # Read args
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic regression data.")
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--n_features', type=int, default=10, help='Number of features to generate')
    parser.add_argument('--noise', type=float, default=0.1, help='Standard deviation of the Gaussian noise')
    parser.add_argument('--no_targets', action='store_false', help='Include target variable in the output')
    args = parser.parse_args()
    
    n_samples = args.n_samples
    n_features = args.n_features
    noise = args.noise
    targets = args.no_targets
    # generate a set of synthetic data
    synthetic_data = generate_synthetic_data(n_samples, n_features, noise, targets)
    # save the synthetic data to a CSV file
    if targets:
        print("Synthetic data generated with target variable.")
        filename = 'synthetic_data_with_target.csv'
    else:
        print("Synthetic data generated without target variable.")
        filename = 'synthetic_data_without_target.csv'
    synthetic_data.to_csv(filename, index=False)
    print(f"Synthetic data generated and saved to '{filename}'.")