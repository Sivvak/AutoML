## load libraries
import openml
import os
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

set_config(transform_output = "pandas")

def run_train_iteration(dataset_id: int, target_column_name: str, model, params: dict):
    ## load dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    print(f'Training started for {type(model).__name__} on dataset {dataset.name}')

    ## define features and target
    X, _, _, _ = dataset.get_data(dataset_format="dataframe")
    y = X.loc[:, target_column_name]
    X = X.drop([target_column_name], axis = 1)

    ## split data into train and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    ## define pipeline
    col_trans = ColumnTransformer(transformers=[
            ('num_pipeline', MinMaxScaler(), make_column_selector(dtype_include=np.number)),
            ('cat_pipeline', OneHotEncoder(handle_unknown='error', sparse_output=False), make_column_selector(dtype_include=['category', np.object_]))
        ],
        remainder='drop',
        n_jobs=-1
    )

    model_pipeline = Pipeline([('preprocessing', col_trans),
                            ('model', model)])

    ## optimize model using grid search
    grid_search = GridSearchCV(model_pipeline, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    ## save results to csv
    path = os.path.join(os.path.expanduser("~\\Desktop"), f'AutoML\\HW1\\results\\{type(model).__name__}\\{dataset.name}.csv')

    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv(path)

    print(f'Training completed for {type(model).__name__} on dataset {dataset.name}')

if __name__ == '__main__':
    datasets = [
        (50, 'Class'),
        (31, 'class'),
        (1464, 'Class'),
        (334, 'class')
    ]

    algorithms = [
        (
            DecisionTreeClassifier(),
            {
                'model__ccp_alpha': [0, 0.2, 0.4, 0.6, 0.8, 1],
                'model__max_depth': [1, 5, 10, 15, 20, 25, 30],
                'model__min_samples_leaf': [1, 10, 20, 30, 40, 50, 60],
                'model__min_samples_split': [2, 10, 20, 30, 40, 50, 60],
            }
        ),
        (
            SVC(),
            [
                {
                    'model__kernel': ['linear'],
                    'model__C': [0.1, 1, 10, 100, 1000]
                },
                {
                    'model__kernel': ['rbf'],
                    'model__C': [0.1, 1, 10, 100, 1000],
                    'model__gamma': [0.001, 0.0001]
                },
                {
                    'model__kernel': ['sigmoid'],
                    'model__C': [0.1, 1, 10, 100, 1000],
                    'model__gamma': [0.001, 0.0001]
                },
                {
                    'model__kernel': ['poly'],
                    'model__C': [0.1, 1, 10, 100, 1000],
                    'model__gamma': [0.001, 0.0001],
                    'model__degree': [2, 3, 4, 5]
                }
            ]
        ),
        (
            GradientBoostingClassifier(),
            {
                'model__learning_rate': [0, 0.2, 0.4, 0.6, 0.8, 1],
                'model__n_estimators': [1, 100, 1000, 2000, 3000, 4000, 5000],
                'model__subsample': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
                'model__max_depth': [1, 5, 10, 15, 20, 25, 30],
            }
        )
    ]

    for dataset_id, target_column_name in datasets:
        for model, params in algorithms:
            run_train_iteration(dataset_id, target_column_name, model, params)

    print('Success')