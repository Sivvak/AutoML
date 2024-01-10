import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from load_data import load_train_data

def filter_chi2(X, y, threshold=0.1):
    _, p = chi2(X, y)
    return X.iloc[:, p < threshold]

if __name__ == '__main__':
    # Load data
    X, y = load_train_data()

    # Apply Chi2
    c2, p = chi2(X, y)

    print((p < 0.05).sum())

    # plot both values on subplots
    c2 = [min(c, 10) for c in c2]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(c2)
    ax[1].plot(p)
    plt.show()
