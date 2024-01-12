from scipy.stats import uniform, randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector as SFS, RFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV, space
from boruta import BorutaPy

def rf(random_state=None):
    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('model', RandomForestClassifier(random_state=random_state)),
        ]
    )


def rf_pca(random_state=None):
    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('pca', PCA(n_components='mle', random_state=random_state)),
            ('model', RandomForestClassifier(random_state=random_state)),
        ]
    )


def rf_sfs(random_state=None):
    rf1 = RandomForestClassifier(random_state=random_state)
    rf2 = RandomForestClassifier(random_state=random_state)

    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('sfs', SFS(rf1, n_features_to_select='auto', tol=1e-3, direction='forward', cv=5, scoring='balanced_accuracy', n_jobs=-1)),
            ('model', rf2),
        ]
    )


def rf_rfe(random_state=None):
    rf1 = RandomForestClassifier(random_state=random_state)
    rf2 = RandomForestClassifier(random_state=random_state)

    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('rfe', RFECV(rf1, min_features_to_select=10, cv=5, scoring='balanced_accuracy', n_jobs=-1)),
            ('model', rf2),
        ]
    )


def rf_tuned(random_state=None):
    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('model', RandomForestClassifier(
                random_state=random_state,
                max_features=0.6280760490974634,
                max_samples=0.9687297765377242,
                n_estimators=188
            )),
        ]
    )


def rf_pca_tuned(random_state=None):
    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('pca', PCA(n_components='mle', random_state=random_state)),
            ('model', RandomForestClassifier(
                random_state=random_state,
                max_features=0.6069480147787453,
                max_samples=0.7259644777835147,
                n_estimators=485
            )),
        ]
    )


def rs_tune_rf(model, X, y, random_state=None):
    rs_params = {
        # 'sfs__estimator__n_estimators': randint(1, 501),
        # 'sfs__estimator__max_samples': uniform(loc=0.1, scale=0.9),
        # 'sfs__estimator__max_features': uniform(loc=0.1, scale=0.9),
        'model__n_estimators': randint(1, 501),
        'model__max_samples': uniform(loc=0.1, scale=0.9),
        'model__max_features': uniform(loc=0.1, scale=0.9)
    }

    rs = RandomizedSearchCV(model, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)
    rs.fit(X, y)

    return rs


def bayes_tune_rf(model, X, y, random_state=None):
    bayes_params = {
        # 'sfs__estimator__n_estimators': space.Integer(1, 500),
        # 'sfs__estimator__max_samples': space.Real(0.1, 1, prior='uniform'),
        # 'sfs__estimator__max_features': space.Real(0.1, 1, prior='uniform'),
        'model__n_estimators': space.Integer(1, 500),
        'model__max_samples': space.Real(0.1, 1, prior='uniform'),
        'model__max_features': space.Real(0.1, 1, prior='uniform')
    }

    bayes = BayesSearchCV(model, bayes_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=3)
    bayes.fit(X, y)

    return bayes


def rf_top_secret_allegro(random_state=None):
    rf1 = RandomForestClassifier(random_state=random_state)
    rf2 = RandomForestClassifier(random_state=random_state)

    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('boruta', BorutaPy(rf1, n_estimators='auto', verbose=2, random_state=random_state)),
            ('model', rf2),
        ]
    )