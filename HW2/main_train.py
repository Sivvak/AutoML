import models
from common import load_train_data
from data_exploration.filter import filter_features
from models import bayes_tune_rf, rs_tune_rf
from sklearn.calibration import column_or_1d
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42

# f_classif - Balanced accuracy 12: 0.85611960035263
#             Balanced accuracy 13: 0.8681494269761975
#             Balanced accuracy 14: 0.8530708198648251

# chi2 - Balanced accuracy 12: 0.8575705260064649
#        Balanced accuracy 13: 0.8620702321481046
#        Balanced accuracy 14: 0.8651190126359095

# load data and labels from files
X, y = load_train_data()
# X = X.iloc[:, filter_features(X, y, random_state)]
y = column_or_1d(y, warn=False)
# X = SelectKBest(f_classif, k=13).fit_transform(X, y)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

# create and fit model
model = models.rf_rfe(random_state)

# tune hyperparams using random search
model.fit(X_train, y_train)
# model = bayes_tune_rf(model, X_train, y_train, random_state)

# evaluate model AUC
y_pred = model.predict(X_test)
print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}')
print(f'Features selected count: {model.named_steps["rfe"].n_features_}')
# print(f'Best params: {model.best_params_}')