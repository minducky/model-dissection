# model-dissection

Three PyTorch model-dissection utilities: benchmark a layer's forward-pass
time (`layer_time`), compute a stack's effective receptive field
(`receptive_field`), and inspect a loaded model's structure and weights
(`model_inside`).

## Usage

```bash
git clone https://github.com/minducky/model-dissection.git
cd model-dissection
pip install -e .
```

```python
from model_dissection import inspect_model, get_value

info = inspect_model(model)
w = get_value(info, idx=5, name="weight")
```

Pick up later changes with:

```bash
git pull
pip install -e .
```
