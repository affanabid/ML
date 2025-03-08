import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Read the dataset
df = pd.read_csv('dataset/manufacturing.csv')
target = df.columns[-1]
features = df.columns[:-1]

# Prepare data
X = df[features]
y = df[target]

def create_poly_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X), poly.get_feature_names_out()

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def backward_elimination(X, y, significance=0.05):
    X_poly, feature_names = create_poly_features(X)
    selected_features = list(range(len(feature_names)))
    
    while len(selected_features) > 1:
        best_r2 = -np.inf
        worst_feature = None
        
        for feature in selected_features:
            current_features = [f for f in selected_features if f != feature]
            model = LinearRegression()
            model.fit(X_poly[:, current_features], y)
            r2 = r2_score(y, model.predict(X_poly[:, current_features]))
            
            if r2 > best_r2:
                best_r2 = r2
                worst_feature = feature
        
        if best_r2 > r2_score(y, LinearRegression().fit(X_poly[:, selected_features], y).predict(X_poly[:, selected_features])):
            selected_features.remove(worst_feature)
        else:
            break
    
    return selected_features, feature_names

def forward_selection(X, y):
    X_poly, feature_names = create_poly_features(X)
    n_features = X_poly.shape[1]
    selected_features = []
    
    while len(selected_features) < n_features:
        best_r2 = -np.inf
        best_feature = None
        
        for feature in range(n_features):
            if feature not in selected_features:
                current_features = selected_features + [feature]
                model = LinearRegression()
                model.fit(X_poly[:, current_features], y)
                r2 = r2_score(y, model.predict(X_poly[:, current_features]))
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_feature = feature
        
        if best_feature is not None:
            if len(selected_features) > 0:
                current_r2 = r2_score(y, LinearRegression().fit(X_poly[:, selected_features], y).predict(X_poly[:, selected_features]))
                if best_r2 - current_r2 < 0.01:  # Stop if improvement is minimal
                    break
            selected_features.append(best_feature)
        else:
            break
    
    return selected_features, feature_names

def bidirectional_selection(X, y):
    X_poly, feature_names = create_poly_features(X)
    n_features = X_poly.shape[1]
    selected_features = []
    
    while len(selected_features) < n_features:
        # Forward step
        best_addition = None
        best_add_r2 = -np.inf
        
        for feature in range(n_features):
            if feature not in selected_features:
                current_features = selected_features + [feature]
                model = LinearRegression()
                model.fit(X_poly[:, current_features], y)
                r2 = r2_score(y, model.predict(X_poly[:, current_features]))
                
                if r2 > best_add_r2:
                    best_add_r2 = r2
                    best_addition = feature
        
        # Backward step
        worst_removal = None
        best_remove_r2 = -np.inf
        
        if len(selected_features) > 1:
            for feature in selected_features:
                current_features = [f for f in selected_features if f != feature]
                model = LinearRegression()
                model.fit(X_poly[:, current_features], y)
                r2 = r2_score(y, model.predict(X_poly[:, current_features]))
                
                if r2 > best_remove_r2:
                    best_remove_r2 = r2
                    worst_removal = feature
        
        # Compare and update
        if best_addition is not None:
            if len(selected_features) > 0:
                current_r2 = r2_score(y, LinearRegression().fit(X_poly[:, selected_features], y).predict(X_poly[:, selected_features]))
                if best_add_r2 - current_r2 < 0.01:  # Stop if improvement is minimal
                    break
            selected_features.append(best_addition)
            
            if worst_removal is not None and worst_removal != best_addition:
                if best_remove_r2 > best_add_r2:
                    selected_features.remove(worst_removal)
        else:
            break
    
    return selected_features, feature_names

def keep_all_variables(X, y):
    X_poly, feature_names = create_poly_features(X)
    return list(range(X_poly.shape[1])), feature_names

def evaluate_model(X, y, selected_features, feature_names, method_name):
    X_poly, _ = create_poly_features(X)
    model = LinearRegression()
    model.fit(X_poly[:, selected_features], y)
    y_pred = model.predict(X_poly[:, selected_features])
    
    r2 = r2_score(y, y_pred)
    adj_r2 = adjusted_r2(r2, len(y), len(selected_features))
    
    print(f"\n{method_name} Results:")
    print(f"Selected features: {[feature_names[i] for i in selected_features]}")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")
    
    return r2, adj_r2

# Apply all methods and compare
methods = {
    "Backward Elimination": backward_elimination,
    "Forward Selection": forward_selection,
    "Bidirectional Selection": bidirectional_selection,
    "Keep All Variables": keep_all_variables
}

results = {}

for method_name, method_func in methods.items():
    selected_features, feature_names = method_func(X, y)
    r2, adj_r2 = evaluate_model(X, y, selected_features, feature_names, method_name)
    results[method_name] = {"R²": r2, "Adjusted R²": adj_r2}

# Plot comparison
plt.figure(figsize=(10, 6))
metrics = pd.DataFrame(results).T
metrics.plot(kind='bar')
plt.title('Comparison of Feature Selection Methods')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('comparison.png')
plt.close()


