import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv('co2.csv')

X = df.drop('CO2 Emissions(g/km)', axis=1)
y = df['CO2 Emissions(g/km)']

numeric_features = ['Engine Size(L)', 'Cylinders', 
                   'Fuel Consumption City (L/100 km)',
                   'Fuel Consumption Hwy (L/100 km)', 
                   'Fuel Consumption Comb (L/100 km)',
                   'Fuel Consumption Comb (mpg)']

categorical_features = ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

numeric_features_transformed = numeric_features
categorical_features_transformed = []
for i, feature in enumerate(categorical_features):
    unique_values = X[feature].nunique()
    categorical_features_transformed.extend([f"{feature}_{j}" for j in range(unique_values-1)])

feature_names = numeric_features_transformed + categorical_features_transformed

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print("\n=== Full Model ===")
full_model = LinearRegression()
full_model.fit(X_train, y_train)

full_model_results = {
    'method': 'No Elimination',
    'selected_features': feature_names,
    'coefficients': dict(zip(feature_names, full_model.coef_)),
    'train_r2': full_model.score(X_train, y_train),
    'test_r2': full_model.score(X_test, y_test)
}

print(f"R² Scores - Train: {full_model_results['train_r2']:.4f}, Test: {full_model_results['test_r2']:.4f}")

def backward_elimination(X, y, feature_names, significance_level=0.05):
    selected_features = list(range(X.shape[1]))
    n_features = len(selected_features)
    
    while len(selected_features) > 0:
        X_with_const = sm.add_constant(X[:, selected_features])
        
        try:
            model = sm.OLS(y, X_with_const).fit()
            
            p_values = model.pvalues[1:]  
            
            max_p_value = p_values.max()
            if max_p_value > significance_level:
                max_p_value_index = p_values.argmax()
                feature_to_remove = selected_features[max_p_value_index]
                selected_features.pop(max_p_value_index)
            else:
                break
                
        except np.linalg.LinAlgError:
            print("Warning: Encountered numerical issues. Stopping elimination process.")
            break
            
        if len(selected_features) == 0 or len(selected_features) < n_features * 0.1:
            break
    
    return selected_features

print("\n=== Backward Elimination Model ===")
selected_features = backward_elimination(X_train, y_train, feature_names)

X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

backward_model = LinearRegression()
backward_model.fit(X_train_selected, y_train)

backward_results = {
    'method': 'Backward Elimination',
    'selected_features': [feature_names[idx] for idx in selected_features],
    'coefficients': dict(zip([feature_names[idx] for idx in selected_features], backward_model.coef_)),
    'train_r2': backward_model.score(X_train_selected, y_train),
    'test_r2': backward_model.score(X_test_selected, y_test)
}

print(f"Features retained: {len(backward_results['selected_features'])}")
print(f"R² Scores - Train: {backward_results['train_r2']:.4f}, Test: {backward_results['test_r2']:.4f}")

def forward_selection(X, y, feature_names, significance_level=0.05):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    
    while remaining_features:
        best_pvalue = float('inf')
        best_feature = None
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_current = X[:, current_features]
            
            try:
                X_with_const = sm.add_constant(X_current)
                
                model = sm.OLS(y, X_with_const).fit()
                
                feature_p_value = model.pvalues.iloc[-1]
                
                if feature_p_value < best_pvalue:
                    best_pvalue = feature_p_value
                    best_feature = feature
                    
            except (np.linalg.LinAlgError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                continue
        
        if best_pvalue < significance_level and best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
            
    return selected_features

print("\n=== Forward Selection Model ===")
forward_selected_features = forward_selection(X_train, y_train, feature_names)

X_train_forward = X_train[:, forward_selected_features]
X_test_forward = X_test[:, forward_selected_features]

forward_model = LinearRegression()
forward_model.fit(X_train_forward, y_train)

forward_results = {
    'method': 'Forward Selection',
    'selected_features': [feature_names[idx] for idx in forward_selected_features],
    'coefficients': dict(zip([feature_names[idx] for idx in forward_selected_features], forward_model.coef_)),
    'train_r2': forward_model.score(X_train_forward, y_train),
    'test_r2': forward_model.score(X_test_forward, y_test)
}

print(f"Features retained: {len(forward_results['selected_features'])}")
print(f"R² Scores - Train: {forward_results['train_r2']:.4f}, Test: {forward_results['test_r2']:.4f}")

def stepwise_selection(X, y, feature_names, significance_level=0.05):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    
    print("Starting stepwise selection process...")
    
    while remaining_features:
        best_pvalue = float('inf')
        best_feature = None
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_current = X[:, current_features]
            
            try:
                X_with_const = sm.add_constant(X_current)
                
                model = sm.OLS(y, X_with_const).fit()
                
                feature_p_value = model.pvalues.iloc[-1]
                
                if feature_p_value < best_pvalue:
                    best_pvalue = feature_p_value
                    best_feature = feature
                    
            except (np.linalg.LinAlgError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                continue
        
        if best_pvalue < significance_level and best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            print(f"Forward: Adding feature {feature_names[best_feature]} with p-value {best_pvalue:.4f}")
            
            while len(selected_features) > 1:
                X_current = X[:, selected_features]
                X_with_const = sm.add_constant(X_current)
                
                try:
                    model = sm.OLS(y, X_with_const).fit()
                    p_values = model.pvalues[1:]  # Exclude constant term
                    
                    max_p_value = p_values.max()
                    max_p_value_index = p_values.argmax()
                    
                    if max_p_value > significance_level:
                        feature_to_remove = selected_features[max_p_value_index]
                        print(f"Backward: Removing feature {feature_names[feature_to_remove]} with p-value {max_p_value:.4f}")
                        
                        selected_features.pop(max_p_value_index)
                        remaining_features.append(feature_to_remove)
                    else:
                        break
                        
                except np.linalg.LinAlgError:
                    break
        else:
            break
            
    return selected_features

print("\n=== Stepwise Selection Model ===")
stepwise_selected_features = stepwise_selection(X_train, y_train, feature_names)

X_train_stepwise = X_train[:, stepwise_selected_features]
X_test_stepwise = X_test[:, stepwise_selected_features]

stepwise_model = LinearRegression()
stepwise_model.fit(X_train_stepwise, y_train)

stepwise_results = {
    'method': 'Stepwise Selection',
    'selected_features': [feature_names[idx] for idx in stepwise_selected_features],
    'coefficients': dict(zip([feature_names[idx] for idx in stepwise_selected_features], stepwise_model.coef_)),
    'train_r2': stepwise_model.score(X_train_stepwise, y_train),
    'test_r2': stepwise_model.score(X_test_stepwise, y_test)
}

print(f"Features retained: {len(stepwise_results['selected_features'])}")
print(f"R² Scores - Train: {stepwise_results['train_r2']:.4f}, Test: {stepwise_results['test_r2']:.4f}")

print("\n=== Model Comparison ===")
print("1. Feature Count Comparison:")
print(f"Full Model Features: {len(full_model_results['selected_features'])}")
print(f"Backward Elimination Features: {len(backward_results['selected_features'])}")
print(f"Forward Selection Features: {len(forward_results['selected_features'])}")
print(f"Stepwise Selection Features: {len(stepwise_results['selected_features'])}")

print("\n2. Performance Comparison:")
print("Full Model:")
print(f"- Training R² Score: {full_model_results['train_r2']:.4f}")
print(f"- Testing R² Score: {full_model_results['test_r2']:.4f}")
print("\nBackward Elimination Model:")
print(f"- Training R² Score: {backward_results['train_r2']:.4f}")
print(f"- Testing R² Score: {backward_results['test_r2']:.4f}")
print("\nForward Selection Model:")
print(f"- Training R² Score: {forward_results['train_r2']:.4f}")
print(f"- Testing R² Score: {forward_results['test_r2']:.4f}")
print("\nStepwise Selection Model:")
print(f"- Training R² Score: {stepwise_results['train_r2']:.4f}")
print(f"- Testing R² Score: {stepwise_results['test_r2']:.4f}")

print("\n=== Detailed Analysis for Handwritten Part ===")

print("\n1. Common Features Across All Selection Methods:")
backward_features = set(backward_results['selected_features'])
forward_features = set(forward_results['selected_features'])
stepwise_features = set(stepwise_results['selected_features'])

common_features = backward_features.intersection(forward_features, stepwise_features)
print("\nFeatures selected by ALL methods:")
for feature in common_features:
    print(f"- {feature}")

print("\nUnique to Backward Elimination:")
for feature in backward_features - (forward_features.union(stepwise_features)):
    print(f"- {feature}")

print("\nUnique to Forward Selection:")
for feature in forward_features - (backward_features.union(stepwise_features)):
    print(f"- {feature}")

print("\nUnique to Stepwise Selection:")
for feature in stepwise_features - (backward_features.union(forward_features)):
    print(f"- {feature}")

print("\n2. Performance Comparison with Full Model:")
print("Difference in R² scores (compared to full model):")
print(f"Backward: Train: {full_model_results['train_r2'] - backward_results['train_r2']:.6f}, "
      f"Test: {full_model_results['test_r2'] - backward_results['test_r2']:.6f}")
print(f"Forward:  Train: {full_model_results['train_r2'] - forward_results['train_r2']:.6f}, "
      f"Test: {full_model_results['test_r2'] - forward_results['test_r2']:.6f}")
print(f"Stepwise: Train: {full_model_results['train_r2'] - stepwise_results['train_r2']:.6f}, "
      f"Test: {full_model_results['test_r2'] - stepwise_results['test_r2']:.6f}")

print("\n3. Coefficient Comparison for Key Features:")
for feature in numeric_features:
    print(f"\nFeature: {feature}")
    print(f"Full Model: {full_model_results['coefficients'][feature]:.4f}")
    if feature in backward_results['coefficients']:
        print(f"Backward:  {backward_results['coefficients'][feature]:.4f}")
    if feature in forward_results['coefficients']:
        print(f"Forward:   {forward_results['coefficients'][feature]:.4f}")
    if feature in stepwise_results['coefficients']:
        print(f"Stepwise:  {stepwise_results['coefficients'][feature]:.4f}")

all_results = {
    'full_model': full_model_results,
    'backward': backward_results,
    'forward': forward_results,
    'stepwise': stepwise_results
}

import json
with open('model_results.json', 'w') as f:
    json.dump(all_results, f, indent=4)

print("\nDetailed results saved to 'model_results.json'")
