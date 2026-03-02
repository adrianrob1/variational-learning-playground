# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Model factory: save/load helpers and architecture registry.

Refactored from tasks/common/models/__init__.py — imports updated to use
vlbench.models.swag, vlbench.models.mcdropout (no longer from parent package).
"""

from typing import Any, Iterable, Mapping
import numpy as np
import torch
from torch import nn
from .models32 import get_model
from .grudense import GRUDense
from .frn import FilterResponseNorm
from .resnet224 import resnet50
from .swag import SWAG
from .mcdropout import MCDropout


def savemodel(
    to,
    modelname: str,
    modelargs: Iterable[Any],
    modelkwargs: Mapping[str, Any],
    model: nn.Module,
    **kwargs,
) -> None:
    """Serialise model weights + constructor metadata to a .pt file.

    Args:
        to: Path to save to.
        modelname: String key (must match a name in globals() of this module).
        modelargs: Positional constructor args.
        modelkwargs: Keyword constructor args.
        model: nn.Module whose state_dict to save.
        **kwargs: Extra keys stored alongside model state.
    """
    dic = {
        "modelname": modelname,
        "modelargs": tuple(modelargs),
        "modelkwargs": {k: modelkwargs[k] for k in modelkwargs},
        "modelstates": model.state_dict(),
        **kwargs,
    }
    torch.save(dic, to)


def loadmodel(fromfile, device=torch.device("cpu")):
    """Load model weights and constructor metadata from a .pt file.

    Args:
        fromfile: Path to the checkpoint file.
        device: Device to load onto.

    Returns:
        (model, extra_dict) where extra_dict holds optimizer/scheduler metadata.
    """
    dic = torch.load(fromfile, map_location=device)
    model = globals()[dic["modelname"]](
        *dic["modelargs"], **dic.get("modelkwargs", {})
    ).to(device)
    model.load_state_dict(dic.pop("modelstates"))
    return model, dic


def resnet20(outclass: int, input_size: int = 32) -> torch.nn.Module:
    """ResNet-20 with Filter Response Normalisation."""
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=torch.nn.Identity,
    )


def resnet20_mcdrop(
    outclass: int, input_size: int = 32, p: float = 0.05
) -> torch.nn.Module:
    """ResNet-20 with MC-Dropout layers."""
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=lambda: MCDropout(p),
    )


def softplus_inv(x: float) -> float:
    return x + np.log(-np.expm1(-x))


def resnet20_bbb(
    outclass: int,
    input_size: int = 32,
    prior_precision: float = 1.0,
    std_init: float = 0.05,
    bnn_type: str = "Reparameterization",
) -> torch.nn.Module:
    """ResNet-20 converted to a Bayes-by-Backprop BNN."""
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

    bnn_options = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0 / np.sqrt(prior_precision),
        "posterior_mu_init": 0.0,
        "posterior_rho_init": softplus_inv(std_init),
        "type": bnn_type,
        "moped_enable": False,
    }
    model = resnet20(outclass, input_size)
    dnn_to_bnn(model, bnn_options)
    return model


def resnet20_swag(outclass: int, input_size: int = 32, max_rank: int = 20) -> SWAG:
    """ResNet-20 wrapped in SWAG."""
    return SWAG(resnet20(outclass, input_size), max_rank)


def preresnet110(outclass: int, input_size: int = 32) -> torch.nn.Module:
    """Pre-activation ResNet-110."""
    return get_model(
        "preresnet110_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=torch.nn.Identity,
    )


def resnet18wide(outclass: int, input_size: int = 32) -> torch.nn.Module:
    """Wide ResNet-18."""
    return get_model(
        "resnet18",
        data_info={"num_classes": outclass, "input_size": input_size},
    )


def densenet121(outclass: int, input_size: int = 32) -> torch.nn.Module:
    """DenseNet-121."""
    return get_model(
        "densenet121",
        data_info={"num_classes": outclass, "input_size": input_size},
    )


def gru_dense(vocab_size: int, num_classes: int, padding_idx: int) -> GRUDense:
    """GRU + Dense classification head."""
    return GRUDense(vocab_size, num_classes, padding_idx)


def resnet50_imagenet(outclass: int, input_size: int = 224) -> torch.nn.Module:
    """ResNet-50 with FRN for ImageNet."""
    return resnet50(
        activation=nn.Identity, norm_layer=FilterResponseNorm, num_classes=outclass
    )


from .bdl_competition import (
    make_uci_mlp,
    make_cifar_alexnet,
    make_medmnist_cnn,
)


def uci_mlp(num_features: int, **kwargs) -> torch.nn.Module:
    """UCI MLP model."""
    return make_uci_mlp({"num_features": num_features}, **kwargs)


def cifar_alexnet(outclass: int, **kwargs) -> torch.nn.Module:
    """CIFAR AlexNet model."""
    return make_cifar_alexnet({"num_classes": outclass}, **kwargs)


def medmnist_lenet(outclass: int, **kwargs) -> torch.nn.Module:
    """MedMNIST LeNet model."""
    return make_medmnist_cnn({"num_classes": outclass}, **kwargs)


STANDARDMODELS = {
    "resnet20": resnet20,
    "resnet18wide": resnet18wide,
    "preresnet110": preresnet110,
    "densenet121": densenet121,
    "resnet50_imagenet": resnet50_imagenet,
    "uci_mlp": uci_mlp,
    "cifar_alexnet": cifar_alexnet,
    "medmnist_lenet": medmnist_lenet,
}
MCDROPMODELS = {"resnet20_mcdrop": resnet20_mcdrop}
BBBMODELS = {"resnet20_bbb": resnet20_bbb}
SWAGMODELS = {"resnet20_swag": resnet20_swag}
MODELS = {**STANDARDMODELS, **MCDROPMODELS, **BBBMODELS, **SWAGMODELS}
