from y_correlation import filter_cor
from information_gain import filter_ig
from chi2 import filter_chi2

def filter_features(X, y):
    X = filter_cor(X, y)
    X = filter_ig(X, y)
    X = filter_chi2(X, y)
    return X

if __name__ == '__main__':
    from load_data import load_train_data
    X, y = load_train_data()
    X = filter_features(X, y)
    print(X.shape)