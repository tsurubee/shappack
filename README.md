# ShapPack

ShapPack is a Python package for interpretable machine learning based on Shapley values.

ShapPack is currently in beta and under active development!

## Installation

```bash
$ pip install shappack
```

## Usage
The usage of ShapPack is almost the same as that of [slundberg/shap](https://github.com/slundberg/shap).
```python
import shappack
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
SEED = 123
np.random.seed(SEED)

# Prepare dataset
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston["data"], boston["target"], test_size=0.2, random_state=SEED)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Prepare model
model = SVR(kernel="rbf")
model.fit(X_train_std, y_train)

# Coumpute SHAP value
i = 2
explainer = shappack.KernelExplainer(model.predict, X_train_std[:100])
shap_value = explainer.shap_values(X_test_std[i], n_workers=-1)
```

For now, ShapPack does not have own visualization mechanism, so it is necessary to use [slundberg/shap](https://github.com/slundberg/shap) for visualization.

```python
import shap
shap.initjs()
shap.force_plot(explainer.base_val[0], shap_value, X_test[i], boston.feature_names)
```

<img src="./docs/images/boston-force-plot.png" alt="boston-force-plot">

An important difference from [slundberg/shap](https://github.com/slundberg/shap) is that the `shap_values` function in ShapPack has three new arguments: `n_workers`, `skip_features`, and `characteristic_func`, which contribute to faster computation and scalability.

The usage of each of these arguments is described below.

### `n_workers`: Multiprocessing

Specify the number of processes used for the calculation of SHAP values.  
`n_workers=-1` means using all processors.  
If the program is running on a multi-core server, we can expect a reduction in computation time.

```python
shap_value = explainer.shap_values(X_test_std[i], n_workers=-1)
```

### `skip_features`: Skip the calculation of SHAP values

We can skips the calculation of SHAP values for the features specified in `skip_features`.
The features to be skipped can be specified by feature name or index number.

Note that we need to pass a list of feature names to KernelExplainer's `feature_names` argument when specifying `skip_features` by feature names.

```python
explainer = shappack.KernelExplainer(model.predict, X_train_std[0:100], feature_names=boston.feature_names)
skip_features=["PTRATIO", "TAX"]
shap_value = explainer.shap_values(X_test_std[i], skip_features=skip_features, n_workers=-1)
feature_names = np.delete(boston.feature_names, explainer.skip_idx)
x_test = np.delete(X_test[i], explainer.skip_idx)
shap.force_plot(explainer.base_val[0], shap_value, x_test, feature_names)
```

### `characteristic_func`: Incorporate own characteristic function

We can incorporate own implemented characteristic functions into the `characteristic_func` argument.

The example below is a function that replaces the expected value calculation in the original Kernel SHAP's characteristic function with a minimum value calculation.

```
def my_characteristic_func(instance, subsets, model, data):
    n_subsets = subsets.shape[0]
    n_data = data.shape[0]
    synth_data = np.tile(data, (n_subsets, 1))
    for i, subset in enumerate(subsets):
        offset = i * n_data
        features_idx = np.where(subset == 1.0)[0]
        synth_data[offset : offset + n_data, features_idx] = instance[:, features_idx][0]
    model_preds = model(synth_data)
    ey = np.zeros(n_subsets)
    for i in range(n_subsets):
        ey[i] = np.min(model_preds[i * n_data : i * n_data + n_data])
    return ey

shap_value = explainer.shap_values(X_test_std[i], characteristic_func=my_characteristic_func, n_workers=-1)
shap.force_plot(explainer.base_val[0], shap_value, X_test[i], boston.feature_names)
```


## License

This project is licensed under the terms of the MIT license, see LICENSE.
