# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import Subset


class ToyDataGenerator:
    """2D binary/multiclass classification with Gaussian blobs for federated learning.

    Ported from bayes-admm/data_generators.py.
    """

    def __init__(
        self,
        setting=1,
        num_clients=5,
        num_samples=2000,
        polynomial_degree=1,
        seed=0,
        multiclass=False,
    ):
        self.state = {
            "seed": seed,
            "setting": setting,
            "num_clients": num_clients,
            "num_samples": num_samples,
            "polynomial_degree": polynomial_degree,
            "multiclass": multiclass,
        }

        if setting == 1:
            self.state["gaussian_centers"] = [
                [[0.0, 0.2], [0.45, 0]],
                [[0.6, 0.9], [0.7, 0.45]],
                [[1.3, 0.4], [1.0, 0.1]],
                [[1.6, -0.1], [1.7, -0.4]],
                [[2.0, 0.3], [2.3, 0.1]],
            ]
            self.state["gaussian_std"] = [
                [[0.08, 0.22], [0.08, 0.16]],
                [[0.24, 0.08], [0.16, 0.08]],
                [[0.04, 0.20], [0.06, 0.16]],
                [[0.16, 0.05], [0.24, 0.05]],
                [[0.05, 0.16], [0.05, 0.22]],
            ]
        elif setting == 2:
            self.state["gaussian_centers"] = [
                [[0.0, 0], [1, 0]],
                [[0.65, 0], [0.85, 0]],
            ]
            self.state["gaussian_std"] = [
                [[0.25, 0.25], [0.25, 0.25]],
                [[0.05, 0.1], [0.05, 0.1]],
            ]
        elif setting == 3:
            self.state["gaussian_centers"] = [[[0.0, 0], [1, 0]], [[0.5, 0], [1.5, 0]]]
            self.state["gaussian_std"] = [
                [[0.2, 0.25], [0.2, 0.25]],
                [[0.2, 0.25], [0.2, 0.25]],
            ]
        else:
            raise ValueError(f"Unknown setting: {setting}")

        if num_clients > len(self.state["gaussian_centers"]):
            raise ValueError(
                f"Setting {setting} supports max {len(self.state['gaussian_centers'])} clients."
            )

        self._create_data()

    def _create_data(self):
        np.random.seed(self.state["seed"])
        self.inputs_plot = []
        self.labels = []
        self.num_points_per_client = []

        for i in range(self.state["num_clients"]):
            X, y = make_blobs(
                self.state["num_samples"] * 2,
                centers=self.state["gaussian_centers"][i],
                cluster_std=self.state["gaussian_std"][i],
                shuffle=False,
            )
            self.inputs_plot.append(X.astype("float32"))
            if self.state["multiclass"]:
                self.labels.append(y + 2 * i)
            else:
                self.labels.append(y)
            self.num_points_per_client.append(len(y))

        self.inputs_plot = np.array(self.inputs_plot)
        self.labels = np.array(self.labels)

        # Range for plotting/testing
        self.x_axis_min, self.x_axis_max = (
            self.inputs_plot[:, :, 0].min() - 0.2,
            self.inputs_plot[:, :, 0].max() + 0.2,
        )
        self.y_axis_min, self.y_axis_max = (
            self.inputs_plot[:, :, 1].min() - 0.2,
            self.inputs_plot[:, :, 1].max() + 0.2,
        )

        h = 0.01
        x_mesh, y_mesh = np.meshgrid(
            np.arange(self.x_axis_min, self.x_axis_max, h),
            np.arange(self.y_axis_min, self.y_axis_max, h),
        )
        self.test_shape = x_mesh.shape
        inputs_test = np.c_[x_mesh.ravel(), y_mesh.ravel()].astype("float32")
        self.inputs_test_plot = torch.from_numpy(inputs_test)
        self.labels_test = torch.zeros(len(self.inputs_test_plot), dtype=torch.float32)

        if self.state["polynomial_degree"] > 1:
            self.poly = PolynomialFeatures(self.state["polynomial_degree"])
            self.inputs = self.poly.fit_transform(
                self.inputs_plot.reshape(-1, 2)
            ).reshape(
                self.state["num_clients"],
                -1,
                self.inputs_plot.shape[-1]
                if self.state["polynomial_degree"] == 1
                else self.poly.n_output_features_,
            )
            # wait, fix shape
            flat_inputs = self.inputs_plot.reshape(-1, 2)
            poly_inputs = self.poly.fit_transform(flat_inputs)
            self.inputs = poly_inputs.reshape(
                self.state["num_clients"], self.state["num_samples"] * 2, -1
            )
            self.inputs_test = torch.from_numpy(
                self.poly.fit_transform(inputs_test).astype("float32")
            )
        else:
            self.inputs = self.inputs_plot
            self.inputs_test = self.inputs_test_plot

        self.num_outputs = np.max(self.labels) + 1 if self.state["multiclass"] else 2
        self.num_parameters = self.inputs.shape[2]

    def data_split(self, client_ind):
        if client_ind >= self.state["num_clients"]:
            raise IndexError("Client index out of range")

        X_train = torch.from_numpy(self.inputs[client_ind].astype("float32"))
        y_train = torch.from_numpy(self.labels[client_ind].astype("int64"))

        return (X_train, y_train), (self.inputs_test, self.labels_test)

    def full_data(self):
        X_train = torch.from_numpy(
            self.inputs.reshape(-1, self.num_parameters).astype("float32")
        )
        y_train = torch.from_numpy(self.labels.reshape(-1).astype("int64"))
        return (X_train, y_train), (self.inputs_test, self.labels_test)
