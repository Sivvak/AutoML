import io
import os 
import pandas as pd
from sklearn.calibration import column_or_1d

main_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Documents', 'Files', 'AutoML', 'HW1', 'repo', 'AutoML', 'HW2')

def load_train_data():
    data_path = os.path.join(os.path.join(main_path, 'data'))
    with open(os.path.join(data_path, 'artificial_train.data')) as table:
        buffer = io.StringIO('\n'.join(line.strip() for line in table))
        X = pd.read_table(buffer, header=None, sep=' ')
    y = pd.read_csv(os.path.join(data_path, 'artificial_train.labels'), header=None)
    # y = column_or_1d(y, warn=False)

    return X, y

def load_test_data():
    data_path = os.path.join(os.path.join(main_path, 'data'))
    with open(os.path.join(data_path, 'artificial_test.data')) as table:
        buffer = io.StringIO('\n'.join(line.strip() for line in table))
        X = pd.read_table(buffer, header=None, sep=' ')

    return X