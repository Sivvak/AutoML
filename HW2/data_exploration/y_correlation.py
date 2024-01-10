import matplotlib.pyplot as plt
from load_data import load_train_data

def filter_cor(X, y, threshold=1e-3):
    cor = X.corrwith(y[0]).to_numpy()
    return abs(cor) > threshold

if __name__ == '__main__':    
    X, y = load_train_data()

    X = filter_cor(X, y)
    
    print(X.shape)

    # cor = X.corrwith(y[0])

    # print((abs(cor) < 1e-3).sum())

    # plt.plot(cor)
    # plt.show()