import os
import warnings
import copy
import itertools
import numpy as np
from scipy.special import binom
from concurrent.futures import ProcessPoolExecutor
from ._base import BaseExplainer
from ..utils._link import convert_to_link
from ..characteristic_funcs.reference import kernel_shap


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

    def shap_values(
        self, X, n_samples="auto", l1_reg="auto", n_workers=1, characteristic_func="kernelshap"
    ):
        X = self.convert_to_nparray(X)
        if len(X.shape) == 1:
            instance = X.reshape(1, -1)
            if instance.shape[1] != self.n_features:
                raise ValueError(
                    "The number of features in instance X and the background dataset do not match."
                )
            shap_values = self._shap_values(
                instance, n_samples, l1_reg, n_workers, characteristic_func
            )
            return shap_values
        elif len(X.shape) == 2:
            # TODO: Support multiple instances
            pass
        else:
            raise ValueError("The instances X to be interpreted must be a vector or 2D matrix")

    def _shap_values(self, instance, n_samples, l1_reg, n_workers, characteristic_func):
        # Compute f(x), the predicted value for instance
        self.fx = self.model(instance)
        # If there is only one feature, it has all the effects
        if self.n_features == 1:
            phi = self.link.f(self.fx) - self.link.f(self.base_val)
            return phi

        # 1. Sampling Subsets (Set of binary vectors)
        self._sampling(n_samples)

        # 2. Applying the Characteristic Function
        if callable(characteristic_func):
            self.characteristic_func = characteristic_func
        elif characteristic_func == "kernelshap":
            self.characteristic_func = kernel_shap
        else:
            raise ValueError('`characteristic_func` should be "kernelshap" or callable function')

        if n_workers == 1:
            self.y_pred = self.characteristic_func(instance, self.subsets, self.model, self.data)
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
            # Splitting data for multi-processing
            subsets_list = np.array_split(self.subsets, n_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        self.characteristic_func, instance, subsets, self.model, self.data
                    )
                    for subsets in subsets_list
                ]
                results = [f.result() for f in futures]
            self.y_pred = np.hstack(results)

            # 3. Solving Weighted Least Squares
        return

    def _sampling(self, n_samples):
        if n_samples == "auto":
            self.n_samples = 2 * self.n_features + 2 ** 11
        else:
            self.n_samples = n_samples
        self.subsets = np.zeros((self.n_samples, self.n_features))
        self.kernel_weights = np.zeros(self.n_samples)
        self.n_added_samples = 0
        n_subset_size = np.int(np.ceil((self.n_features - 1) / 2.0))
        n_paired_subset_size = np.int(np.floor((self.n_features - 1) / 2.0))
        weight_vector = np.array(
            [
                (self.n_features - 1.0) / (i * (self.n_features - i))
                for i in range(1, n_subset_size + 1)
            ]
        )
        weight_vector[:n_paired_subset_size] *= 2
        weight_vector /= np.sum(weight_vector)
        remaining_weight_vector = copy.copy(weight_vector)
        n_remaining_samples = self.n_samples
        n_full_subsets = 0
        binary_vec = np.zeros(self.n_features)
        for subset_size in range(1, n_subset_size + 1):
            n_sampling_subsets = binom(self.n_features, subset_size)
            if subset_size <= n_paired_subset_size:
                n_sampling_subsets *= 2
            # If there are enough number of remaining samples to sample n_sampling_subsets
            if n_remaining_samples * remaining_weight_vector[subset_size - 1] > n_sampling_subsets:
                n_full_subsets += 1
                n_remaining_samples -= n_sampling_subsets
                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]
                weight = weight_vector[subset_size - 1] / binom(self.n_features, subset_size)
                if subset_size <= n_paired_subset_size:
                    weight /= 2.0
                for idx in itertools.combinations(
                    np.arange(self.n_features, dtype="int64"), subset_size
                ):
                    binary_vec[:] = 0.0
                    binary_vec[np.array(idx, dtype="int64")] = 1.0
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
                binary_vec[np.random.permutation(self.n_features)[:subset_size]] = 1.0

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

    def _add_sample(self, binary_vec, weight):
        self.subsets[self.n_added_samples, :] = binary_vec
        self.kernel_weights[self.n_added_samples] = weight
        self.n_added_samples += 1

    def _solve(self):
        pass
