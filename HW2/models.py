from scipy.stats import uniform, randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def rf(random_state=None):
    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('model', RandomForestClassifier(random_state=random_state)),
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


def rf_pca(random_state=None):
    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('pca', PCA(n_components=480, random_state=random_state)),
            ('model', RandomForestClassifier(random_state=random_state)),
        ]
    )


def rf_pca_tuned(random_state=None):
    return Pipeline(
        [
            ('preprocessing', MinMaxScaler()),
            ('pca', PCA(n_components=480, random_state=random_state)),
            ('model', RandomForestClassifier(
                random_state=random_state,
                max_features=0.6069480147787453,
                max_samples=0.7259644777835147,
                n_estimators=485
            )),
        ]
    )


def tune_rf(model, X, y, random_state):
    rs_params = {
        'model__n_estimators': randint(1, 501),
        'model__max_samples': uniform(loc=0.1, scale=0.9),
        'model__max_features': uniform(loc=0.1, scale=0.9)
    }

    rs = RandomizedSearchCV(model, rs_params, n_iter=50, cv=5, scoring='balanced_accuracy', n_jobs=-1, random_state=random_state, verbose=2)
    rs.fit(X, y)

    return rs