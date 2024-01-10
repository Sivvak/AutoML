# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from HW1.repo.AutoML.HW2.data_exploration.load_data import load_train_data
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

X, y = load_train_data()

print(X.shape)
print(y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

chi_square_value,p_value = calculate_bartlett_sphericity(X)
print(f'Bartlett\'s test:\n\tChi-squared: {chi_square_value}, p-value: {p_value}')

kmo_all,kmo_model = calculate_kmo(X)
print(f'KMO test:\n\tKMO: {kmo_model}')