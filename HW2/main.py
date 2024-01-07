import io
import os 
import numpy as np
import pandas as pd

# load data and labels from files
data_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'AutoML', 'HW2', 'data')

with open(os.path.join(data_path, 'artificial_train.data')) as table:
    buffer = io.StringIO('\n'.join(line.strip() for line in table))
    df = pd.read_table(buffer, header=None, sep=' ')

X = df.drop(df.columns[-1], axis=1)
y = pd.read_csv(os.path.join(data_path, 'artificial_train.labels'), header=None)