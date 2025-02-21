import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

def load_data():
    df = pd.read_csv('dataset/Real estate.csv')
    
    feature_names = [
        'transaction_date',
        'house_age',
        'distance_to_mrt',
        'num_convenience_stores',
        'latitude',
        'longitude'
    ]
    
    df.columns = ['No'] + feature_names + ['house_price']
    
    X = df[feature_names]
    y = df['house_price']
    
    return X, y, feature_names

def experiment_split_sizes():
    X, y, _ = load_data()
    split_sizes = [0.7, 0.8, 0.9]
    results = []
    
    for train_size in split_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        results.append({
            'train_size': train_size,
            'train_score': train_score,
            'test_score': test_score
        })
    
    return pd.DataFrame(results)

def experiment_random_seeds():

    X, y, _ = load_data()
    seeds = [42, 123, 456, 789, 101112]
    results = []
    
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=seed
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        results.append({
            'seed': seed,
            'train_score': train_score,
            'test_score': test_score
        })
    
    return pd.DataFrame(results)

def experiment_dataset_sizes():
    X, y, _ = load_data()
    data_sizes = [0.25, 0.5, 0.75, 1.0]
    seeds = [42, 123, 456, 789, 101112]  # Multiple seeds
    results = []
    
    for size in data_sizes:
        size_results = []
        for seed in seeds:
            if size < 1.0:
                X_subset, _, y_subset, _ = train_test_split(
                    X, y, train_size=size, random_state=seed
                )
            else:
                X_subset, y_subset = X, y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y_subset, train_size=0.8, random_state=seed
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            size_results.append({
                'train_score': train_score,
                'test_score': test_score
            })
        
        avg_train = np.mean([r['train_score'] for r in size_results])
        avg_test = np.mean([r['test_score'] for r in size_results])
        
        results.append({
            'data_size': size,
            'train_score': avg_train,
            'test_score': avg_test,
            'n_samples': len(X_subset)
        })
    
    return pd.DataFrame(results)

def plot_experiments():
    split_results = experiment_split_sizes()
    plt.figure(figsize=(10, 6))
    plt.plot(split_results['train_size'], split_results['train_score'], 'o-', label='Train Score')
    plt.plot(split_results['train_size'], split_results['test_score'], 'o-', label='Test Score')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('R² Score')
    plt.title('Model Performance vs Training Set Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/split_size_results.png')
    plt.close()
    
    seed_results = experiment_random_seeds()
    plt.figure(figsize=(10, 6))
    plt.plot(seed_results['seed'], seed_results['train_score'], 'o-', label='Train Score')
    plt.plot(seed_results['seed'], seed_results['test_score'], 'o-', label='Test Score')
    plt.xlabel('Random Seed')
    plt.ylabel('R² Score')
    plt.title('Model Performance vs Random Seed')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/random_seed_results.png')
    plt.close()
    
    size_results = experiment_dataset_sizes()
    plt.figure(figsize=(10, 6))
    plt.plot(size_results['data_size'], size_results['train_score'], 'o-', label='Train Score')
    plt.plot(size_results['data_size'], size_results['test_score'], 'o-', label='Test Score')
    plt.xlabel('Dataset Size (%)')
    plt.ylabel('R² Score')
    plt.title('Model Performance vs Dataset Size (Averaged over 5 seeds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/dataset_size_results.png')
    plt.close()
    
    return split_results, seed_results, size_results

def main():

    split_results, seed_results, size_results = plot_experiments()
    
    print("\nSplit Size Experiment Results:")
    print(split_results)
    
    print("\nRandom Seed Experiment Results:")
    print(seed_results)
    
    print("\nDataset Size Experiment Results:")
    print(size_results)
main()
