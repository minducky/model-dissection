"""Inspect a torch.nn.Module's structure and look up its parameters/buffers.

Typical usage::

    model = MyModelClass(...)
    model.load_state_dict(torch.load("ckpt.pth"))

    info = inspect_model(model, input_shape=(1, 3, 224, 224))  # prints the table
    w = get_value(info, idx=5, name="weight")                  # fetch a tensor
    act = get_intermediate_output(model, info, x, idx=5)        # fetch an activation
"""

from typing import Any

import torch
import torch.nn as nn

_INDENT_UNIT = 2  # spaces per nesting level


def _is_container(module: nn.Module) -> bool:
    return len(list(module.named_children())) > 0


def _leaf_values(module: nn.Module) -> dict[str, tuple[str, torch.Tensor]]:
    """Return {name: (kind, tensor)} for a leaf module's params and buffers."""
    values = {}
    for name, tensor in module.named_parameters(recurse=False):
        values[name] = ("Param", tensor)
    for name, tensor in module.named_buffers(recurse=False):
        values[name] = ("Buffer", tensor)
    return values


def _container_labels(model: nn.Module) -> dict[int, tuple[str, int]]:
    """Map id(container) -> (class_name, sibling_index) from the static tree.

    Sibling index counts same-class containers among the children of the
    same immediate parent (e.g. the two BasicBlocks in a Sequential get
    sibling_index 0 and 1). This is a property of the model's definition, not
    of how many times a module is actually called, so it's computed once
    without running forward.
    """
    labels: dict[int, tuple[str, int]] = {}

    def walk(module: nn.Module) -> None:
        type_counters: dict[str, int] = {}
        for _, child in module.named_children():
            if _is_container(child):
                cls_name = type(child).__name__
                sibling_index = type_counters.get(cls_name, 0)
                type_counters[cls_name] = sibling_index + 1
                labels[id(child)] = (cls_name, sibling_index)
            walk(child)

    walk(model)
    return labels


def _trace_forward(
    model: nn.Module,
    x: torch.Tensor,
    container_labels: dict[int, tuple[str, int]],
) -> list[dict[str, Any]]:
    """Run one forward pass and record each container entry/leaf call in order.

    A leaf module invoked more than once per forward (e.g. a shared
    `self.relu`) produces one trace entry per call, each with its own
    occurrence count, rather than being collapsed into a single entry.
    """
    trace: list[dict[str, Any]] = []
    container_stack: list[nn.Module] = []
    occurrence_counter: dict[int, int] = {}
    path_of = {id(m): name for name, m in model.named_modules() if name}
    handles = []

    def container_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        cls_name, sibling_index = container_labels[id(module)]
        trace.append({
            "kind": "container",
            "class_name": cls_name,
            "sibling_index": sibling_index,
            "ancestor_count": len(container_stack),
        })
        container_stack.append(module)

    def container_post_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        container_stack.pop()

    def leaf_post_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        mod_id = id(module)
        occurrence = occurrence_counter.get(mod_id, 0)
        occurrence_counter[mod_id] = occurrence + 1
        path = path_of[mod_id]
        trace.append({
            "kind": "leaf",
            "path": path,
            "name": path.rsplit(".", 1)[-1],
            "class_name": type(module).__name__,
            "ancestor_count": len(container_stack),
            "values": _leaf_values(module),
            "module": module,
            "occurrence": occurrence,
        })

    for mod_name, mod in model.named_modules():
        if not mod_name:
            continue
        if _is_container(mod):
            handles.append(mod.register_forward_pre_hook(container_pre_hook))
            handles.append(mod.register_forward_hook(container_post_hook))
        else:
            handles.append(mod.register_forward_hook(leaf_post_hook))

    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()

    return trace


def inspect_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Trace one forward pass and print the model's execution structure.

    Runs `model(torch.randn(*input_shape))` once (in eval mode, no_grad) and
    records every module call in the order it actually happens. A leaf module
    invoked more than once per forward (e.g. a shared `self.relu`) gets one
    row per call rather than being collapsed into a single entry.

    Args:
        model: A loaded nn.Module instance (weights already applied).
        input_shape: Shape for a random example input used to trace the model.
        model_name: Label to print as the model's name. Defaults to the
            model's class name.

    Returns:
        A dict with "model_name" and "entries" (idx -> {"path", "class_name",
        "values": {name: tensor}}), suitable for passing to get_value() and
        get_intermediate_output().
    """
    model_name = model_name or type(model).__name__
    x = torch.randn(*input_shape)
    container_labels = _container_labels(model)
    trace = _trace_forward(model, x, container_labels)

    root_values = _leaf_values(model)
    nodes: list[dict[str, Any]] = []
    if root_values:
        nodes.append({
            "kind": "leaf",
            "path": "(root)",
            "name": "(root)",
            "class_name": type(model).__name__,
            "ancestor_count": 0,
            "values": root_values,
            "module": model,
            "occurrence": 0,
        })
    nodes.extend(trace)

    # Pre-render each leaf's rows as (node, idx_cell, calc, type_, name_, shape_).
    rendered_rows = []
    idx = 0
    for node in nodes:
        if node["kind"] != "leaf":
            continue
        indent = " " * (node["ancestor_count"] * _INDENT_UNIT)
        idx_content = f"{indent}{idx}"
        calc = f"{node['name']} ({node['class_name']})"
        values = node["values"]
        if values:
            first = True
            for value_name, (kind, tensor) in values.items():
                shape_str = str(list(tensor.shape))
                rendered_rows.append((
                    node, idx_content if first else "", calc if first else "",
                    kind, value_name, shape_str,
                ))
                first = False
        else:
            rendered_rows.append((node, idx_content, calc, "-", "-", "-"))
        node["idx"] = idx
        idx += 1

    idx_col_w = max([len("idx")] + [len(r[1]) for r in rendered_rows])
    calc_col_w = max([len("calc")] + [len(r[2]) for r in rendered_rows])
    type_col_w = max([len("type")] + [len(r[3]) for r in rendered_rows])
    name_col_w = max([len("name")] + [len(r[4]) for r in rendered_rows])
    shape_col_w = max([len("shape")] + [len(r[5]) for r in rendered_rows])

    def _row(idx_cell, calc_cell, type_cell, name_cell, shape_cell):
        return (
            f" {idx_cell.ljust(idx_col_w)} | {calc_cell.ljust(calc_col_w)} | "
            f"{type_cell.ljust(type_col_w)} | {name_cell.ljust(name_col_w)} | "
            f"{shape_cell.ljust(shape_col_w)}"
        )

    header = _row("idx", "calc", "type", "name", "shape")
    separator = "-" * len(header)

    print(f"Model Name : {model_name}")
    print(separator)
    print(header)
    print(separator)

    row_iter = iter(rendered_rows)
    pending_row = next(row_iter, None)
    for node in nodes:
        if node["kind"] == "container":
            if node["ancestor_count"] == 0:
                print()
                print(f"{node['class_name']} [{node['sibling_index']}]")
            else:
                indent = " " * ((node["ancestor_count"] - 1) * _INDENT_UNIT + 1)
                print(f"{indent}└─ {node['class_name']} [{node['sibling_index']}]")
            continue
        # leaf: emit all its rendered rows in order
        while pending_row is not None and pending_row[0] is node:
            _, idx_cell, calc_cell, type_cell, name_cell, shape_cell = pending_row
            print(_row(idx_cell, calc_cell, type_cell, name_cell, shape_cell))
            pending_row = next(row_iter, None)

    print(separator)

    entries = {
        node["idx"]: {
            "path": node["path"],
            "class_name": node["class_name"],
            "values": {name: tensor for name, (_, tensor) in node["values"].items()},
            "module": node["module"],
            "occurrence": node["occurrence"],
        }
        for node in nodes
        if node["kind"] == "leaf"
    }
    return {"model_name": model_name, "entries": entries}


def get_value(info: dict[str, Any], idx: int, name: str | None = None) -> Any:
    """Look up a Param/Buffer tensor from the structure returned by inspect_model.

    Args:
        info: Return value of inspect_model().
        idx: The layer index shown in the printed table.
        name: Parameter/buffer name (as shown in the "name" column). If
            omitted, returns all of that layer's values as a dict.

    Returns:
        The requested tensor, or a {name: tensor} dict if `name` is omitted.
    """
    entries = info["entries"]
    if idx not in entries:
        raise KeyError(f"idx {idx} not found. Available idx: {sorted(entries.keys())}")
    entry = entries[idx]
    if name is None:
        return dict(entry["values"])
    if name not in entry["values"]:
        raise KeyError(
            f"name '{name}' not found at idx {idx} ({entry['path']}). "
            f"Available names: {list(entry['values'].keys())}"
        )
    return entry["values"][name]


def get_intermediate_output(
    model: nn.Module, info: dict[str, Any], x: torch.Tensor, idx: int
) -> Any:
    """Run a full forward pass and capture the activation produced at idx.

    The model is run end-to-end exactly as it normally would be - nothing is
    truncated at `idx`. A forward hook captures the output of the specific
    call idx refers to, which matters when the same module (e.g. a shared
    `self.relu`) is invoked more than once per forward: each call got its own
    idx in inspect_model(), and only the matching occurrence is captured here.

    Args:
        model: The same model instance that `info` was built from.
        info: Return value of inspect_model().
        x: Input tensor to run through the model.
        idx: The layer index shown in the printed table.

    Returns:
        The output produced at that idx during this forward pass (a tensor,
        or whatever that module's forward() returns).
    """
    entries = info["entries"]
    if idx not in entries:
        raise KeyError(f"idx {idx} not found. Available idx: {sorted(entries.keys())}")
    entry = entries[idx]
    target_module = entry["module"]
    target_occurrence = entry["occurrence"]

    captured: dict[str, Any] = {}
    call_count = 0

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal call_count
        if call_count == target_occurrence:
            captured["output"] = output
        call_count += 1

    handle = target_module.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()

    return captured["output"]


if __name__ == "__main__":
    class BasicBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + x)

    class DemoResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            # Same channel count both stages: this demo skips the downsample
            # projection real ResNets use when channel count changes.
            self.layer1 = nn.Sequential(BasicBlock(64), BasicBlock(64))
            self.layer2 = nn.Sequential(BasicBlock(64), BasicBlock(64))

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    demo_model = DemoResNet()
    info = inspect_model(demo_model, input_shape=(1, 3, 32, 32), model_name="DemoResNet")

    print()
    print(
        "get_value(info, idx=1, name='weight') : ",
        get_value(info, idx=1, name="weight").shape,
    )
    print(
        "get_value(info, idx=1, name='running_mean') : ",
        get_value(info, idx=1, name="running_mean").shape,
    )
    print("get_value(info, idx=1) keys : ", list(get_value(info, idx=1).keys()))

    # layer1's first BasicBlock.relu is called twice per forward (once after
    # bn1, once on the residual sum) - each call got its own idx above.
    relu_idxs = [
        idx
        for idx, entry in info["entries"].items()
        if entry["class_name"] == "ReLU" and entry["path"] == "layer1.0.relu"
    ]
    print()
    print(f"layer1.0.relu idxs (2 calls, same module): {relu_idxs}")

    demo_x = torch.randn(1, 3, 32, 32)
    first_call_out = get_intermediate_output(demo_model, info, demo_x, idx=relu_idxs[0])
    second_call_out = get_intermediate_output(demo_model, info, demo_x, idx=relu_idxs[1])
    print(f"get_intermediate_output idx={relu_idxs[0]} shape : {first_call_out.shape}")
    print(f"get_intermediate_output idx={relu_idxs[1]} shape : {second_call_out.shape}")
    print(f"same tensor? {torch.equal(first_call_out, second_call_out)}")
