import numpy as np
from typing import Any


def kernel_shap(
    instance: np.ndarray, subsets: np.ndarray, model: Any, data: np.ndarray
) -> np.ndarray:
    n_subsets = subsets.shape[0]
    n_data = data.shape[0]
    synth_data = np.tile(data, (n_subsets, 1))
    for i, subset in enumerate(subsets):
        offset = i * n_data
        features_idx = np.where(subset == 1.0)[0]
        synth_data[offset : offset + n_data, features_idx] = instance[:, features_idx][0]
    model_preds = model(synth_data)
    # Computing the expected value of the model's predictions for the background dataset
    if len(model_preds.shape) == 1:
        ey = np.zeros(n_subsets)
        for i in range(n_subsets):
            ey[i] = np.mean(model_preds[i * n_data : i * n_data + n_data])
    else:
        out_dim = model_preds.shape[1]
        ey = np.zeros((n_subsets, out_dim))
        for i in range(n_subsets):
            ey[i, :] = np.mean(model_preds[i * n_data : i * n_data + n_data, :], 0)
    return ey
