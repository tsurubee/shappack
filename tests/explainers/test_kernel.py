import shappack
import numpy as np
from typing import Tuple


def test_shap_values_regression(
    boston_data: Tuple[np.ndarray, np.ndarray, np.ndarray, "sklearn.svm.SVR"]
) -> None:
    X_train, X_test, y_train, model = boston_data
    explainer = shappack.KernelExplainer(model.predict, X_train[0:10])
    shap_value = explainer.shap_values(X_test[1])
    assert len(shap_value) == X_train.shape[1]
    assert np.round(np.sum(shap_value), 2) == np.round(explainer.fx - explainer.base_val, 2)


def test_shap_values_classification(
    iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, "sklearn.svm.SVC"]
) -> None:
    X_train, X_test, y_train, model = iris_data
    explainer = shappack.KernelExplainer(model.predict_proba, X_train, link="logit")
    shap_value = explainer.shap_values(X_test[0], n_samples=100)
    assert shap_value.shape[0] == explainer.out_dim
    assert shap_value.shape[1] == X_train.shape[1]


def test_shap_values_multiprocessing(
    boston_data: Tuple[np.ndarray, np.ndarray, np.ndarray, "sklearn.svm.SVR"]
) -> None:
    X_train, X_test, y_train, model = boston_data
    explainer = shappack.KernelExplainer(model.predict, X_train[0:10])
    shap_value = explainer.shap_values(X_test[15], n_workers=-1)
    assert len(shap_value) == X_train.shape[1]
    assert np.round(np.sum(shap_value), 2) == np.round(explainer.fx - explainer.base_val, 2)


def test_shap_values_own_characteristic_func(
    boston_data: Tuple[np.ndarray, np.ndarray, np.ndarray, "sklearn.svm.SVR"]
) -> None:
    X_train, X_test, y_train, model = boston_data
    explainer = shappack.KernelExplainer(model.predict, X_train[0:10])

    def my_characteristic_func(instance, subsets, model, data):
        return np.ones(subsets.shape[0])

    shap_value = explainer.shap_values(X_test[1], characteristic_func=my_characteristic_func)
    assert len(shap_value) == X_train.shape[1]
