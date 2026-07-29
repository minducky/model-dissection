"""Compute the effective receptive field of a stack of conv/pooling layers."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def cal_RF(layer_params: Sequence[tuple[Any, Any, Any]]) -> dict[str, list[np.ndarray]]:
    """Compute the accumulated receptive field after each layer.

    Each layer (convolution or pooling, no distinction needed) contributes to
    the receptive field according to its kernel size, stride, and dilation:

        RF_i   = RF_(i-1) + (kernel_i - 1) * dilation_i * jump_(i-1)
        jump_i = jump_(i-1) * stride_i

    starting from RF_0 = 1 and jump_0 = 1 before the first layer.

    Args:
        layer_params: Sequence of ``(kernel_size, stride, dilation)`` tuples,
            one per layer, in forward-pass order. Each element may be a
            scalar (same value for every spatial dimension) or an array-like
            of per-dimension values (e.g. ``(kh, kw)``).

    Returns:
        A dict with ``"RF"`` and ``"jump"``, each a list of numpy arrays
        holding the running receptive field / stride accumulation after every
        layer.
    """
    history = {"RF": [], "jump": []}
    n_layer = len(layer_params)

    RF = None
    jump = None
    for layer_idx, (kernel, stride, dilation) in enumerate(layer_params):
        kernel = np.array(kernel)
        stride = np.array(stride)
        dilation = np.array(dilation)

        if layer_idx == 0:
            print(f"dimension : {kernel.shape}")
            RF = np.ones_like(kernel)
            jump = np.ones_like(kernel)

        RF = RF + (kernel - 1) * dilation * jump
        jump = jump * stride

        history["RF"].append(RF.copy())
        history["jump"].append(jump.copy())

    rf_width = 15
    jump_width = 15

    print("---------------------------- Summary ----------------------------")
    print("Layer   | Receptive Field | Stride Accumulation")
    print("========|=================|======================================")
    for layer_idx in range(n_layer):
        rf_str = str(history["RF"][layer_idx].tolist())
        rf_str = rf_str + " " * (rf_width - len(rf_str))
        jump_str = str(history["jump"][layer_idx].tolist())
        jump_str = jump_str + " " * (jump_width - len(jump_str))
        print(f"{str(layer_idx).zfill(2)}th    | {rf_str} | {jump_str}")
        if layer_idx == n_layer - 1:
            print("=================================================================")
            print(f"Effective filter size : {rf_str}")

    print("-----------------------------------------------------------------")
    return history


if __name__ == "__main__":
    """
    Set per-layer (kernel_size, stride, dilation), in forward-pass order.
    Example: conv(k=3,s=1,d=1) -> conv(k=3,s=1,d=2) -> maxpool(k=2,s=2,d=1)
    """
    layer_params = [
        (3, 1, 2),
        (3, 1, 2),
        (2, 2, 1),
    ]

    cal_RF(layer_params)
