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

    def shap_values(self, X, n_samples="auto"):
        pass

    def _sampling(self, n_samples):
        pass
