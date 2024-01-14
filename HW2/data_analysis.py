import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from common import load_train_data

def plot_importances(X, y):
    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
    importance = np.abs(ridge.coef_).reshape(-1)
    plt.bar(height=importance, x=np.arange(importance.shape[0]))
    plt.title("Feature importances via coefficients")
    plt.show()

if __name__ == '__main__':
    # Load data
    X, y = load_train_data()

    # Apply Information Gain
    plot_importances(X, y)