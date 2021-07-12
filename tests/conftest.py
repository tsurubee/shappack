import pytest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 123

@pytest.fixture()
def boston_data():
    np.random.seed(SEED)
    boston = load_boston()
    X_train, X_test, y_train, _ = train_test_split(boston["data"], boston["target"], test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    model = SVR()
    model.fit(X_train, y_train)
    return (X_train, X_test, y_train, model)
