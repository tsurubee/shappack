import os
import warnings
import copy
import itertools
import numpy as np
from scipy.special import binom
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Union, Callable, Optional
from .base import BaseExplainer
from ..utils.link import convert_to_link
from ..utils.docker import get_cpu_quota_within_docker
from ..characteristic_funcs.reference import kernel_shap


class KernelExplainer(BaseExplainer):
    """Using Kernel SHAP to interpret the output of a machine learning model

    The model-agnostic Kernel SHAP estimates the SHAP value for any model using weighted linear regression.
    For more information on Kernel SHAP, see the original paper below.
    S. Lundberg and S. I. Lee, A Unified Approach to Interpreting Model Predictions, Advances in
    Neural Information Processing Systems 30(NIPS 2017), 2017.

    Args:
        model:
            Prediction function of the machine learning model
        data:
            Background dataset. How to use the background dataset depends on the implementation of
            the characteristic function. In the original Kernel SHAP, the values of the background dataset
            are replaced as references to simulate "missing" features.
        link:
            A generalized linear model link to connect the feature importance values to the model output
    """

    def __init__(
        self,
        model: Any,
        data: Union[List, np.ndarray],
        link: str = "identity",
        feature_names: Union[List, np.ndarray] = None,
    ) -> None:
        self.model = model
        self.data = self.convert_to_nparray(data)
        self.link = convert_to_link(link)
        self.linkf = np.vectorize(self.link.f)
        self.n_data = self.data.shape[0]
        self.n_features = self.data.shape[1]
        if feature_names is not None:
            feature_names = self.convert_to_nparray(feature_names)
            if len(feature_names) != self.n_features:
                raise ValueError(
                    "The length of `feature_names` must be the same as the number of features"
                )
            self.feature_names = feature_names

        if self.n_data > 100:
            warnings.warn(
                "Using "
                + str(self.n_data)
                + " background data samples could cause slower run times.",
                UserWarning,
            )

        # Compute base value (=E[f(x)]=phi_0)
        out_val = np.squeeze(self.model(self.data))
        self.base_val = self.linkf(np.sum(out_val, 0) / self.n_data)
        if len(self.base_val.shape) == 0:
            self.base_val = np.array([self.base_val])

        if len(out_val.shape) == 0 or 1:
            self.out_dim = 1
        elif len(out_val.shape) == 2:
            self.out_dim = out_val.shape[1]
        else:
            raise ValueError("The output of model must be a vector or two-dimensional matrix")

    def shap_values(
        self,
        X: Union[List, np.ndarray],
        n_samples: Union[str, int] = "auto",
        l1_reg: Union[str, int] = "auto",
        n_workers: int = 1,
        characteristic_func: Union[
            str, Callable[[np.ndarray, np.ndarray, Any, np.ndarray], np.ndarray]
        ] = "kernelshap",
        skip_features: Optional[List[Union[str, int]]] = None,
    ) -> np.ndarray:
        """Compute SHAP values

        Args:
            X:
                Data to be interpreted
            n_samples:
                Number of subsets to be sampled. The default value of "auto" uses
                `n_samples = 2 * (Number of features) + 2048`.
            l1_reg:
                The l1 regularization to use for feature selection.
            n_workers:
                Number of processes used in the computation.
                `-1` means using all processors.
            characteristic_func:
               Function that returns the output of the model for each subset.
               In the default case, the original function of Kernel SHAP is executed.
               Users can incorporate their own functions.
            skip_features:
               If there are features that skips the calculation of the SHAP value,
               specify its index or name.

        Returns:
            Array of calculated SHAP values

        """
        X = self.convert_to_nparray(X)
        if len(X.shape) == 1:
            instance = X.reshape(1, -1)
            if instance.shape[1] != self.n_features:
                raise ValueError(
                    "The number of features in instance X and the background dataset do not match."
                )
            shap_values = self._shap_values(
                instance, n_samples, l1_reg, n_workers, characteristic_func, skip_features
            )
            return shap_values
        elif len(X.shape) == 2:
            # TODO: Support multiple instances
            pass
        else:
            raise ValueError(
                "The instances X to be interpreted must be a vector or two-dimensional matrix"
            )

    def _shap_values(
        self,
        instance: np.ndarray,
        n_samples: Union[str, int],
        l1_reg: Union[str, int],
        n_workers: int,
        characteristic_func: Union[
            str, Callable[[np.ndarray, np.ndarray, Any, np.ndarray], np.ndarray]
        ],
        skip_features: Optional[List[Union[str, int]]],
    ) -> np.ndarray:
        # Compute f(x), the predicted value for instance
        self.fx = np.squeeze(self.model(instance))
        if len(self.fx.shape) == 0:
            self.fx = np.array([self.fx])
        # If there is only one feature, it has all the effects
        if self.n_features == 1:
            phi = self.link.f(self.fx) - self.link.f(self.base_val)
            return phi

        self.n_skip_features = 0
        if skip_features is not None:
            self.n_skip_features = len(skip_features)
            self.skip_idx = np.zeros(self.n_skip_features, dtype=int)
            for i, feature in enumerate(skip_features):
                if isinstance(feature, str):
                    self.skip_idx[i] = np.where(self.feature_names == feature)[0][0]
                elif isinstance(feature, int):
                    self.skip_idx[i] = feature
                else:
                    raise ValueError("The elements of `skip_features` must be `str` or `int`.")
            self.skip_idx = np.sort(self.skip_idx)
        self.n_features_calc = self.n_features - self.n_skip_features

        # 1. Sampling Subsets (Set of binary vectors)
        self._sampling(n_samples)

        # 2. Applying the Characteristic Function
        if callable(characteristic_func):
            self.characteristic_func = characteristic_func
        elif characteristic_func == "kernelshap":
            self.characteristic_func = kernel_shap
        else:
            raise ValueError('`characteristic_func` should be "kernelshap" or callable function')

        # If `skip_features` is specified, reconstruct subsets with the features to be skipped set to 1
        # for input to the model.
        subsets_model = self.subsets
        if self.n_skip_features >= 1:
            for i in self.skip_idx:
                subsets_model = np.insert(subsets_model, i, 1, axis=1)

        if n_workers == 1:
            self.y_pred = self.characteristic_func(instance, subsets_model, self.model, self.data)
            if self.out_dim == 1:
                self.y_pred = self.y_pred.reshape((-1, 1))
        else:
            cpu_count = os.cpu_count()
            if n_workers < -1:
                raise ValueError("n_workers option should be -1 or non-negative value")
            elif n_workers == -1:
                n_workers = cpu_count or 1
            elif n_workers > cpu_count:
                warnings.warn(
                    f"n_workers={n_workers} is larger than os.cpu_count()={cpu_count}.",
                    UserWarning,
                )
                n_workers = cpu_count or 1
            # In order to handle the case of running in a Docker container,
            # check the cgroup configuration for CPU limits
            cpu_count_docker = get_cpu_quota_within_docker()
            if cpu_count_docker is not None and cpu_count_docker < n_workers:
                warnings.warn(
                    f"n_workers={n_workers} is larger than Docker CPU limits={cpu_count_docker}.",
                    UserWarning,
                )
                n_workers = cpu_count_docker
            # Splitting data for multi-processing
            subsets_list = np.array_split(subsets_model, n_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        self.characteristic_func, instance, subsets, self.model, self.data
                    )
                    for subsets in subsets_list
                ]
                results = [f.result() for f in futures]
            if self.out_dim == 1:
                self.y_pred = np.hstack(results).reshape((-1, 1))
            else:
                self.y_pred = np.vstack(results)

        # 3. Solving Weighted Least Squares
        phi = np.zeros((self.out_dim, self.n_features_calc))
        for dim in range(self.out_dim):
            phi[dim, :] = self._solve(dim, l1_reg)

        return np.squeeze(phi)

    def _sampling(self, n_samples: Union[str, int]) -> None:
        if n_samples == "auto":
            self.n_samples = 2 * self.n_features_calc + 2 ** 11
        else:
            self.n_samples = n_samples
        max_n_samples = 2 ** self.n_features_calc - 2
        if self.n_samples > max_n_samples:
            self.n_samples = max_n_samples
        self.subsets = np.zeros((self.n_samples, self.n_features_calc), dtype=np.int8)
        self.kernel_weights = np.zeros(self.n_samples)
        self.n_added_samples = 0
        n_subset_size = np.int(np.ceil((self.n_features_calc - 1) / 2.0))
        n_paired_subset_size = np.int(np.floor((self.n_features_calc - 1) / 2.0))
        weight_vector = np.array(
            [
                (self.n_features_calc - 1.0) / (i * (self.n_features_calc - i))
                for i in range(1, n_subset_size + 1)
            ]
        )
        weight_vector[:n_paired_subset_size] *= 2
        weight_vector /= np.sum(weight_vector)
        remaining_weight_vector = copy.copy(weight_vector)
        n_remaining_samples = self.n_samples
        n_full_subsets = 0
        binary_vec = np.zeros(self.n_features_calc, dtype=np.int8)
        for subset_size in range(1, n_subset_size + 1):
            n_sampling_subsets = binom(self.n_features_calc, subset_size)
            if subset_size <= n_paired_subset_size:
                n_sampling_subsets *= 2
            # If there are enough number of remaining samples to sample n_sampling_subsets
            if (
                n_remaining_samples * remaining_weight_vector[subset_size - 1] / n_sampling_subsets
                >= 1.0 - 1e-8
            ):
                n_full_subsets += 1
                n_remaining_samples -= n_sampling_subsets
                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]
                weight = weight_vector[subset_size - 1] / binom(self.n_features_calc, subset_size)
                if subset_size <= n_paired_subset_size:
                    weight /= 2.0
                for idx in itertools.combinations(
                    np.arange(self.n_features_calc, dtype="int64"), subset_size
                ):
                    binary_vec[:] = 0
                    binary_vec[np.array(idx, dtype="int64")] = 1
                    self._add_sample(binary_vec, weight)
                    # Create a binary vector with inverted "0" and "1".
                    if subset_size <= n_paired_subset_size:
                        binary_vec[:] = np.abs(binary_vec - 1)
                        self._add_sample(binary_vec, weight)
            else:
                break
        # Add samples to the rest of the subset space
        n_fixed_samples = self.n_added_samples
        if n_full_subsets != n_subset_size:
            remaining_weight_vector = copy.copy(weight_vector)
            remaining_weight_vector[:n_paired_subset_size] /= 2
            remaining_weight_vector = remaining_weight_vector[n_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)
            idx_set = np.random.choice(
                len(remaining_weight_vector),
                size=4 * int(n_remaining_samples),
                p=remaining_weight_vector,
            )
            idx_set_pos = 0
            used_binary_vec = {}
            while n_remaining_samples > 0 and idx_set_pos < len(idx_set):
                binary_vec[:] = 0.0
                idx = idx_set[idx_set_pos]
                idx_set_pos += 1
                subset_size = idx + n_full_subsets + 1
                binary_vec[np.random.permutation(self.n_features_calc)[:subset_size]] = 1.0

                # only add the sample if we have not seen it before, otherwise just
                # increment a previous sample's weight
                binary_tuple = tuple(binary_vec)
                is_new_sample = False
                if binary_tuple not in used_binary_vec:
                    is_new_sample = True
                    used_binary_vec[binary_tuple] = self.n_added_samples
                    n_remaining_samples -= 1
                    self._add_sample(binary_vec, 1.0)
                else:
                    self.kernel_weights[used_binary_vec[binary_tuple]] += 1.0

                if n_remaining_samples > 0 and subset_size <= n_paired_subset_size:
                    binary_vec[:] = np.abs(binary_vec - 1)
                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    if is_new_sample:
                        n_remaining_samples -= 1
                        self._add_sample(binary_vec, 1.0)
                    else:
                        # we know the compliment sample is the next one after the original sample, so + 1
                        self.kernel_weights[used_binary_vec[binary_tuple] + 1] += 1.0

            # Normalize the kernel weights for the random samples to equal the weight left after
            # the fixed enumerated samples have been already counted
            weight_left = np.sum(weight_vector[n_full_subsets:])
            self.kernel_weights[n_fixed_samples:] *= (
                weight_left / self.kernel_weights[n_fixed_samples:].sum()
            )

    def _add_sample(self, binary_vec: np.ndarray, weight: float) -> None:
        self.subsets[self.n_added_samples, :] = binary_vec
        self.kernel_weights[self.n_added_samples] = weight
        self.n_added_samples += 1

    def _solve(self, dim: int, l1_reg: Union[str, int]) -> np.ndarray:
        # TODO: Support Lasso model and l1_reg argument
        ey_diff = self.linkf(self.y_pred[:, dim]) - self.base_val[dim]
        nonzero_inds = np.arange(self.n_features_calc)
        if len(nonzero_inds) == 0:
            return np.zeros(self.n_features_calc)

        # Eliminate one variable with the constraint that all features sum to the output
        ey_diff2 = ey_diff - self.subsets[:, nonzero_inds[-1]] * (
            self.link.f(self.fx[dim]) - self.base_val[dim]
        )
        etmp = np.transpose(
            np.transpose(self.subsets[:, nonzero_inds[:-1]]) - self.subsets[:, nonzero_inds[-1]]
        )

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernel_weights))
        etmp_dot = np.dot(np.transpose(tmp), etmp)
        try:
            tmp2 = np.linalg.inv(etmp_dot)
        except np.linalg.LinAlgError:
            tmp2 = np.linalg.pinv(etmp_dot)
        w = np.dot(tmp2, np.dot(np.transpose(tmp), ey_diff2))
        phi = np.zeros(self.n_features_calc)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.base_val[dim]) - sum(w)

        # clean up any rounding errors
        for i in range(self.n_features_calc):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi
