import io
import os
import numpy as np
import pandas as pd
from common import main_path
from datetime import datetime
from scipy.stats import uniform, randint
from sklearn.calibration import column_or_1d
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

random_state = 42

# load data and labels from files
with open(os.path.join(main_path, 'data', 'artificial_train.data')) as table:
    buffer = io.StringIO('\n'.join(line.strip() for line in table))
    X = pd.read_table(buffer, header=None, sep=' ')

y = pd.read_csv(os.path.join(main_path, 'data', 'artificial_train.labels'), header=None)
y = column_or_1d(y, warn=False)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# define pipeline
rf1 = RandomForestClassifier(random_state=random_state)
rf2 = RandomForestClassifier(random_state=random_state)

sfs = SFS(rf1, n_features_to_select=480, direction='backward', cv=5, scoring='balanced_accuracy', n_jobs=-1)

rs = pipeline = Pipeline(
    [
        ('preprocessing', MinMaxScaler()),
        ('sfs', sfs),
        # ('pca', PCA(n_components=480, random_state=random_state)),
        ('model', rf2),
        # ('model', TabPFNClassifier(device='cpu', subsample_features=True, N_ensemble_configurations=3))
    ]
)

# tune hyperparams using random search
# rs_params = {
#         'model__n_estimators': randint(1, 501),
#         'model__max_samples': uniform(loc=0.1, scale=0.9),
#         'model__max_features': uniform(loc=0.1, scale=0.9)
#     }

# rs = RandomizedSearchCV(pipeline, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=2)
rs.fit(X_train, y_train)

# evaluate model AUC
y_pred = rs.predict(X_test)
print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}')
print(f'Selected features: {rs.get_feature_names_out()}')
print(f'Current time: {datetime.now().strftime("%H:%M:%S")}')

# max_features=0.6280760490974634, max_samples=0.9687297765377242, n_estimators=188 BEZ PCA
# max_features=0.6069480147787453, max_samples=0.7259644777835147, n_estimators=485 Z PCA