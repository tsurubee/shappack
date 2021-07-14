import numpy as np
from typing import List, Union


class BaseExplainer(object):
    def __init__(self) -> None:
        pass

    def convert_to_nparray(self, data: Union[List, np.ndarray]) -> np.ndarray:
        """Convert to np.ndarray"""
        if not isinstance(data, np.ndarray):
            return np.array(data)
        else:
            return data

    def shap_values(self, *args):
        """Placeholder intended to be overwritten by individual models."""
        raise NotImplementedError
