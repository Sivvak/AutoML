import os
import numpy as np
import models
from common import load_test_data, load_train_data, main_path
from data_exploration.filter import filter_features
from datetime import datetime
from models import bayes_tune_rf, rs_tune_rf
from sklearn.calibration import column_or_1d
from sklearn.feature_selection import SelectKBest, f_classif

random_state = 42

# load data and labels from files
X_train, y_train = load_train_data()
y_train = column_or_1d(y_train, warn=False)

feature_selector = SelectKBest(f_classif, k=13).fit(X_train, y_train)

X_train = feature_selector.transform(X_train)

X_test = load_test_data()
X_test = feature_selector.transform(X_test)

# create and fit model
model = models.rf(random_state)

# tune hyperparams using random search
# model.fit(X_train, y_train)
model = bayes_tune_rf(model, X_train, y_train, random_state)
print(f'Best params: {model.best_params_}')
print(f'Best score: {model.best_score_}')

# save predict probabilities with default formatter
y_pred_proba = model.predict_proba(X_test)[:, 1]
np.savetxt(os.path.join(main_path, 'output', f'313450_313472_artifical_model_prediction_{datetime.now().strftime("%H_%M_%S")}.txt'), y_pred_proba, header='313450_313472', comments='', fmt='%s')