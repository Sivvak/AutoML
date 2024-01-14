import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import column_or_1d
from sklearn.feature_selection import mutual_info_regression
from load_data import load_train_data

def filter_ig(X, y, random_state, threshold=1e-5):
    y = column_or_1d(y, warn=False)
    ig = mutual_info_regression(X, y, random_state = random_state)
    return ig > threshold

if __name__ == '__main__':
    # Load data
    X, y = load_train_data()

    # Apply Information Gain
    ig = mutual_info_regression(X, y)

    print((ig < 1e-5).sum())

    # plt.plot(ig)
    # plt.show()