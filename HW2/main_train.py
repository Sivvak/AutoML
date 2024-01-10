import models
from common import load_train_data
from models import tune_rf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42

# load data and labels from files
X, y = load_train_data()

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

# create and fit model
model = models.rf_tuned(random_state)

# tune hyperparams using random search
model.fit(X_train.iloc[:, :480], y_train)
# model = tune_rf(model, X_train, y_train, random_state)

# evaluate model AUC
y_pred = model.predict(X_test.iloc[:, :480])
print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}')