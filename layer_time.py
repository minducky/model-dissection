"""Measure the forward-pass time of a torch.nn.Module."""

import time

import torch


def measure_layer_time(model, input_shape, device="cpu", n_warmup=10, n_iter=100):
    """Measure forward pass time of a torch.nn.Module in milliseconds.

    Args:
        model: The module to benchmark.
        input_shape: Shape tuple used to generate a random input tensor via
            ``torch.randn(*input_shape)``.
        device: Device to run the benchmark on.
        n_warmup: Number of warmup iterations before timing starts.
        n_iter: Number of timed iterations.

    Returns:
        Average forward pass time in milliseconds.
    """
    model = model.to(device)
    x = torch.randn(*input_shape).to(device)
    model.eval()

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start.record()
            for _ in range(n_iter):
                model(x)
            end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / n_iter
    else:
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(n_iter):
                model(x)
            end = time.perf_counter()
        elapsed = (end - start) * 1000 / n_iter

    print(f"[{device}] {elapsed:.3f} ms / forward")
    return elapsed


if __name__ == "__main__":
    """
    Set layer/model & input_shape
    """
    model = torch.nn.Linear(512, 256)
    input_shape = (32, 512)

    measure_layer_time(model, input_shape, device="cpu")

    if torch.cuda.is_available():
        measure_layer_time(model, input_shape, device="cuda")
    else:
        print("CUDA not available, skipping GPU benchmark")
