import os
import numpy as np
from common import load_test_data, load_train_data, main_path

# load data and labels from files
X_train, y_train = load_train_data()
X_test = load_test_data()

# create and fit model
model = train_model(X_train, y_train)

# save predict probabilities with default formatter
y_pred_proba = model.predict_proba(X_test)[:, 1]
np.savetxt(os.path.join(main_path, 'output', '313450_313472_artifical_model_prediction.txt'), y_pred_proba, header='313450_313472', comments='', fmt='%s')