# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any, Dict


def plot_2d_federated(
    model_outputs_list: List[torch.Tensor],
    toy_generator: Any,
    worker_indices: List[int] = [-1],
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot 2D visualization for federated toy data.

    Ported from bayes-admm/utils.py.
    """
    if toy_generator.num_outputs == 2:
        colours = ["#A9CCE0", "#AB5735"]
        imshow_values = [1 / 24, 19 / 24]
    else:
        colours = [
            "#A9CCE0",
            "#3774B3",
            "#B0E28C",
            "#3AA23D",
            "#F69698",
            "#E50028",
            "#F8BF73",
            "#FB7D21",
            "#C5AFD5",
            "#6B3897",
            "#FDFE9F",
            "#AB5735",
        ]
        imshow_values = [i / 24 for i in range(1, 21, 2)]

    num_plots = len(model_outputs_list)
    plt.figure(figsize=(9 * num_plots, 9))

    for count, outputs in enumerate(model_outputs_list):
        if num_plots > 1:
            current_worker_indices = [worker_indices[count]]
        else:
            current_worker_indices = worker_indices

        _, class_preds = torch.max(outputs, dim=-1)
        # Map class indices to imshow values for colormap
        mapped_preds = np.array([imshow_values[p] for p in class_preds])
        mapped_preds = mapped_preds.reshape(toy_generator.test_shape)

        ax = plt.subplot(1, num_plots, count + 1)
        ax.set_title(f"{title} Node {current_worker_indices}")

        plt.imshow(
            mapped_preds,
            cmap="Paired",
            extent=(
                toy_generator.x_axis_min,
                toy_generator.x_axis_max,
                toy_generator.y_axis_min,
                toy_generator.y_axis_max,
            ),
            vmin=0,
            vmax=1,
            origin="lower",
        )

        # Plot training points for these workers
        for worker_idx in current_worker_indices:
            if worker_idx == -1:  # Global model (show all data)
                inputs = toy_generator.inputs_plot.reshape(-1, 2)
                labels = toy_generator.labels.reshape(-1)
            else:
                inputs = toy_generator.inputs_plot[worker_idx]
                labels = toy_generator.labels[worker_idx]

            unique_labels = np.unique(labels)
            for label in unique_labels:
                idx = np.where(labels == label)
                plt.scatter(
                    inputs[idx, 0],
                    inputs[idx, 1],
                    facecolors="none",
                    edgecolors=colours[label % len(colours)],
                    s=30,
                    linewidth=2,
                    alpha=0.6,
                )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_convergence(
    metrics: Dict[str, List[float]],
    title: str = "Convergence",
    y_label: str = "Metric",
    save_path: Optional[str] = None,
):
    """Plot convergence curves for different federated methods."""
    plt.figure(figsize=(10, 6))
    for label, values in metrics.items():
        plt.plot(values, label=label, linewidth=2)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Communication Round")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
