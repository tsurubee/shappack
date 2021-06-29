import warnings
import numpy as np
from ._base import BaseExplainer
from ..utils._link import convert_to_link


class KernelExplainer(BaseExplainer):
    def __init__(self, model, data, link="identity"):
        self.model = model
        self.data = self.convert_to_nparray(data)
        self.link = convert_to_link(link)
        self.linkf = np.vectorize(self.link.f)
        self.n_data = self.data.shape[0]
        self.n_features = self.data.shape[1]

        if self.n_data > 100:
            warnings.warn(
                "Using "
                + str(len(self.n_data))
                + " background data samples could cause slower run times.",
                UserWarning,
            )

        # Compute base value (=E[f(x)]=phi_0)
        out_val = self.model(self.data)
        self.base_val = np.sum(self.linkf(out_val)) / self.n_data
        # TODO: Support classification problem outputs

    def shap_values(self, X, n_samples="auto", l1_reg="auto"):
        X = self.convert_to_nparray(X)
        if len(X.shape) == 1:
            instance = X.reshape(1, -1)
            if instance.shape[1] != self.n_features:
                raise ValueError(
                    "The number of features in instance X and the background dataset do not match."
                )
            shap_values = self._shap_values(instance, n_samples)
            return shap_values
        elif len(X.shape) == 2:
            # TODO: Support multiple instances
            pass
        else:
            raise ValueError("The instances X to be interpreted must be a vector or 2D matrix")

    def _shap_values(self, instance, n_samples, l1_reg):
        # Sampling Subsets (Set of binary vectors)

        # Applying the Characteristic Function

        # Solving Weighted Least Squares
        pass

    def _sampling(self, n_samples):
        pass

    def _solve(self):
        pass
