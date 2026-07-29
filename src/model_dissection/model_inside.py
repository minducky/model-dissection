"""Inspect a torch.nn.Module's structure and look up its parameters/buffers.

Typical usage::

    model = MyModelClass(...)
    model.load_state_dict(torch.load("ckpt.pth"))

    info = inspect_model(model)                # prints the structure table
    w = get_value(info, idx=5, name="weight")  # fetch a specific tensor
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


def _build_tree(
    module: nn.Module, path: str = "", ancestor_count: int = 0
) -> list[dict[str, Any]]:
    """Recursively walk `module` into a flat list of container/leaf nodes."""
    nodes = []
    type_counters = {}
    for name, child in module.named_children():
        child_path = f"{path}.{name}" if path else name
        cls_name = type(child).__name__
        if _is_container(child):
            sibling_index = type_counters.get(cls_name, 0)
            type_counters[cls_name] = sibling_index + 1
            nodes.append({
                "kind": "container",
                "class_name": cls_name,
                "sibling_index": sibling_index,
                "ancestor_count": ancestor_count,
            })
            nodes.extend(_build_tree(child, child_path, ancestor_count + 1))
        else:
            nodes.append({
                "kind": "leaf",
                "path": child_path,
                "name": name,
                "class_name": cls_name,
                "ancestor_count": ancestor_count,
                "values": _leaf_values(child),
            })
    return nodes


def inspect_model(model: nn.Module, model_name: str | None = None) -> dict[str, Any]:
    """Print the model's structure and return a lookup structure for get_value.

    Args:
        model: A loaded nn.Module instance (weights already applied).
        model_name: Label to print as the model's name. Defaults to the
            model's class name.

    Returns:
        A dict with "model_name" and "entries" (idx -> {"path", "class_name",
        "values": {name: tensor}}), suitable for passing to get_value().
    """
    model_name = model_name or type(model).__name__
    root_values = _leaf_values(model)
    nodes = []
    if root_values:
        nodes.append({
            "kind": "leaf",
            "path": "(root)",
            "name": "(root)",
            "class_name": type(model).__name__,
            "ancestor_count": 0,
            "values": root_values,
        })
    nodes.extend(_build_tree(model))

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
            self.layer1 = nn.Sequential(BasicBlock(64), BasicBlock(64))
            self.layer2 = nn.Sequential(BasicBlock(128), BasicBlock(128))

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    demo_model = DemoResNet()
    info = inspect_model(demo_model, model_name="DemoResNet")

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
