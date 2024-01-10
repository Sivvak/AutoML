import io
import os
import numpy as np
import pandas as pd
from common import main_path
from sklearn.calibration import column_or_1d

# load data and labels from files
with open(os.path.join(main_path, 'data', 'artificial_train.data')) as table:
    buffer = io.StringIO('\n'.join(line.strip() for line in table))
    X_train = pd.read_table(buffer, header=None, sep=' ')

with open(os.path.join(main_path, 'data', 'artificial_test.data')) as table:
    buffer = io.StringIO('\n'.join(line.strip() for line in table))
    X_test = pd.read_table(buffer, header=None, sep=' ')

y_train = pd.read_csv(os.path.join(main_path, 'data', 'artificial_train.labels'), header=None)
y_train = column_or_1d(y, warn=False)

# create and fit model
model = train_model(X_train, y_train)

# save predict probabilities with default formatter
y_pred_proba = model.predict_proba(X_test.iloc[:, :480])[:, 1]
np.savetxt(os.path.join(main_path, 'output', '313450_313472_artifical_model_prediction.txt'), y_pred_proba, header='313450_313472', comments='', fmt='%s')