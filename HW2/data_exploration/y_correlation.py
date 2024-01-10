import matplotlib.pyplot as plt
from load_data import load_train_data

X, y = load_train_data()

cor = X.corrwith(y[0])

print((abs(cor) < 1e-3).sum())

plt.plot(cor)
plt.show()