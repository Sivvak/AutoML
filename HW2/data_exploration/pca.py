from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from HW1.repo.AutoML.HW2.data_exploration.load_data import load_train_data

X, y = load_train_data()

scaler = StandardScaler()
pca = PCA()

X = scaler.fit_transform(X)

principalComponents = pca.fit_transform(X)

print(principalComponents)
sum = pca.explained_variance_ratio_.sum()

plt.plot(np.cumsum(pca.explained_variance_ratio_) / sum)
plt.show()