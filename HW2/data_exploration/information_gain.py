import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from load_data import load_train_data

def filter_ig(X, y, threshold=1e-5):
    ig = mutual_info_regression(X, y)
    return X.iloc[:, ig > threshold]

if __name__ == '__main__':
    # Load data
    X, y = load_train_data()

    # Apply Information Gain
    ig = mutual_info_regression(X, y)

    print((ig < 1e-5).sum())

    # plt.plot(ig)
    # plt.show()