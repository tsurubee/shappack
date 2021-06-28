import numpy as np


class BaseExplainer(object):
    def __init__(self):
        pass

    def convert_to_nparray(self, data):
        if not isinstance(data, np.ndarray):
            return np.array(data)
        else:
            return data

    def shap_values(self, *args):
        """
        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError
