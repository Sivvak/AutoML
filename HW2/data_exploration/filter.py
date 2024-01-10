from y_correlation import filter_cor
from information_gain import filter_ig
from chi2 import filter_chi2

def filter_features(X, y):
    cor_mask = filter_cor(X, y)
    ig_mask = filter_ig(X, y)
    chi2_mask = filter_chi2(X, y)
    return [cor and ig and chi for cor, ig, chi in zip(cor_mask, ig_mask, chi2_mask)]

if __name__ == '__main__':
    from load_data import load_train_data
    X, y = load_train_data()
    X = X.iloc[:, filter_features(X, y)]
    print(X.shape)