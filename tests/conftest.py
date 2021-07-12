import pytest
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC


@pytest.fixture()
def boston_data():
    boston = load_boston()
    X_train, X_test, y_train, _ = train_test_split(boston["data"], boston["target"], test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    model = SVR()
    model.fit(X_train, y_train)
    return (X_train, X_test, y_train, model)

@pytest.fixture()
def iris_data():
    iris = load_iris()
    X_train, X_test, y_train, _ = train_test_split(iris["data"], iris["target"], test_size=0.2)
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)
    return (X_train, X_test, y_train, model)

