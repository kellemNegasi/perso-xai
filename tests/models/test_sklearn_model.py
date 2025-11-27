import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.models.sklearn_models import SklearnModel


def test_sklearn_model_fit_predict_and_attribute_passthrough():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=0,
    )
    estimator = LogisticRegression(max_iter=200, random_state=0)
    model = SklearnModel(name="logreg", estimator=estimator)

    model.fit(X, y)
    preds_model = model.predict(X)
    preds_direct = estimator.predict(X)
    assert np.array_equal(preds_model, preds_direct)

    proba_model = model.predict_proba(X[:5])
    proba_direct = estimator.predict_proba(X[:5])
    assert np.allclose(proba_model, proba_direct)

    # __getattr__ should proxy trained attributes such as coef_
    assert np.allclose(model.coef_, estimator.coef_)


def test_sklearn_model_without_predict_proba_raises():
    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_informative=4,
        random_state=1,
    )
    estimator = LinearSVC(random_state=1)
    model = SklearnModel(name="linear_svc", estimator=estimator)

    model.fit(X, y)
    assert model.supports_proba is False
    with pytest.raises(AttributeError):
        model.predict_proba(X[:3])


def test_sklearn_model_handles_string_labels_and_predict_numeric():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        random_state=2,
    )
    y_str = np.where(y == 1, ">50K", "<=50K")
    estimator = LogisticRegression(max_iter=200, random_state=2)
    model = SklearnModel(name="logreg_strings", estimator=estimator)

    model.fit(X, y_str)

    preds = model.predict(X[:5])
    assert set(preds) <= {"<=50K", ">50K"}

    numeric_preds = model.predict_numeric(X[:5])
    assert numeric_preds.dtype == float
    assert set(np.unique(numeric_preds)).issubset({0.0, 1.0})
