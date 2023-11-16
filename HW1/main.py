## load libraries
import openml
import os
import sys
import numpy as np
import pandas as pd
from bayes_initial_params import get_bayes_initial_params
from scipy.stats import uniform, randint
from sklearn import set_config
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV, space

set_config(transform_output = "pandas")

hw1_path = os.path.join(os.path.expanduser("~"), f'Desktop\\AutoML\\HW1')

run_random = '--random' in sys.argv
run_bayes = '--bayes' in sys.argv

sample_size = 10
iterations_count = 50

def run_train_iteration(dataset_id: int, target_column_name: str, model, random_search_params: dict, bayes_params: dict):
    ## load dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    print(f'Training started for {type(model).__name__} on dataset {dataset.name}')

    ## define features and target
    X, _, _, _ = dataset.get_data(dataset_format="dataframe")
    y = X.loc[:, target_column_name]
    X = X.drop([target_column_name], axis = 1)

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

    if run_random:
        ## tune hyperparams using grid search
        grid_search = GridSearchCV(model_pipeline, random_search_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X, y)

        ## save results to csv
        path = os.path.join(hw1_path, f'results\\{type(model).__name__}\\random\\{dataset.name}.csv')

        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv(path)

    if run_bayes:
        ## tune hyperparams using bayesian optimization
        bayes_search = BayesSearchCV(model_pipeline, bayes_params, n_iter=iterations_count, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        bayes_search.fit(X, y)

        ## save results to csv
        path = os.path.join(hw1_path, f'results\\{type(model).__name__}\\bayes\\{dataset.name}.csv')

        results = pd.DataFrame(bayes_search.cv_results_)
        results.to_csv(path)

    print(f'Training completed for {type(model).__name__} on dataset {dataset.name}')


if __name__ == '__main__':
    datasets = [
        (50, 'Class'),
        (31, 'class'),
        (1464, 'Class'),
        (334, 'class')
    ]

    svc_C = [10 ** x for x in uniform(loc=-10, scale=13).rvs(sample_size)]
    svc_gamma = [10 ** x for x in uniform(loc=-10, scale=13).rvs(sample_size)]

    algorithms = [
        (
            DecisionTreeClassifier(),
            {
                'model__ccp_alpha': uniform().rvs(sample_size),
                'model__max_depth': randint(1, 31).rvs(sample_size),
                'model__min_samples_leaf': randint(1, 61).rvs(sample_size),
                'model__min_samples_split': randint(2, 61).rvs(sample_size)
            },
            {
                'model__ccp_alpha': space.Real(0, 1, prior='uniform'),
                'model__max_depth': space.Integer(1, 30),
                'model__min_samples_leaf': space.Integer(1, 60),
                'model__min_samples_split': space.Integer(2, 60)
            }
        ),
        (
            SVC(),
            [
                {
                    'model__kernel': ['linear'],
                    'model__C': svc_C
                },
                {
                    'model__kernel': ['rbf'],
                    'model__C': svc_C,
                    'model__gamma': svc_gamma
                },
                {
                    'model__kernel': ['sigmoid'],
                    'model__C': svc_C,
                    'model__gamma': svc_gamma
                },
                {
                    'model__kernel': ['poly'],
                    'model__C': svc_C,
                    'model__gamma': svc_gamma,
                    'model__degree': randint(2, 6).rvs(sample_size)
                }
            ],
            [
                {
                    'model__kernel': ['linear'],
                    'model__C': space.Real(1e-10, 1e3, prior='log-uniform')
                },
                {
                    'model__kernel': ['rbf'],
                    'model__C': space.Real(1e-10, 1e3, prior='log-uniform'),
                    'model__gamma': space.Real(1e-10, 1e3, prior='log-uniform')
                },
                {
                    'model__kernel': ['sigmoid'],
                    'model__C': space.Real(1e-10, 1e3, prior='log-uniform'),
                    'model__gamma': space.Real(1e-10, 1e3, prior='log-uniform')
                },
                {
                    'model__kernel': ['poly'],
                    'model__C': space.Real(1e-10, 1e3, prior='log-uniform'),
                    'model__gamma': space.Real(1e-10, 1e3, prior='log-uniform'),
                    'model__degree': space.Integer(2, 5)
                }
            ]
        ),
        (
            GradientBoostingClassifier(),
            {
                'model__learning_rate': [10 ** x for x in uniform(loc=-10, scale=10).rvs(sample_size)],
                'model__n_estimators': randint(1, 5001).rvs(sample_size),
                'model__subsample': uniform(loc=0.1, scale=0.9).rvs(sample_size),
                'model__max_depth': randint(1, 16).rvs(sample_size),
            },
            {
                'model__learning_rate': space.Real(1e-10, 1, prior='log-uniform'),
                'model__n_estimators': space.Integer(1, 5000),
                'model__subsample': space.Real(0.1, 1, prior='uniform'),
                'model__max_depth': space.Integer(1, 15)
            }
        )
    ]

    ## tune hyperparameters
    for dataset_id, target_column_name in datasets:
        for model, random_search_params, bayes_params in algorithms:
            run_train_iteration(dataset_id, target_column_name, model, random_search_params, bayes_params)

    print('Success')