---
tags: [mlops, session1, serialization, onnx, msgpack]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 04 · Serialization Formats & ONNX

> [!info] Source of truth
> Rewritten directly from `msgpack_example.py`, `pytorch_to_onnx.py`, and the README's ONNX section — all code below is copied from the actual files, not reconstructed.

## JSON vs MessagePack — the actual demo (`msgpack_example.py`)
```python
import json
import msgpack

data = {"distance_km": 5.2, "passengers": 2, "model_version": "v1.3"}

# JSON — text format
json_bytes = json.dumps(data).encode()
print(len(json_bytes))   # 62 bytes
print(json_bytes)        # b'{"distance_km": 5.2, "passengers": 2, ...}'

# MessagePack — binary format
msgpack_bytes = msgpack.packb(data)
print(len(msgpack_bytes))   # 53 bytes  (~15% smaller)
print(msgpack_bytes)        # b'\x83\xabdistance_km...'  (unreadable, but smaller)
```
- 62 → 53 bytes is ~15% smaller **on this small payload**. The README notes the gap **widens on larger, number-heavy payloads** — this specific example undersells MessagePack's real advantage at scale.
- MessagePack output is genuinely unreadable when printed — that's the trade: smaller/faster in exchange for losing human-readability.

### The second half of the same file — Redis caching (I missed this entirely first pass)
```python
import redis
import msgpack

r = redis.Redis()

features = {"user_age": 28, "purchase_history": [12.5, 8.0, 45.2], "last_login_days": 3}

try:
    # Writing features to Redis with MessagePack
    r.set("user:1234:features", msgpack.packb(features))

    # Reading and deserializing for prediction
    raw = r.get("user:1234:features")
    features = msgpack.unpackb(raw, raw=False)
    print(features)
    # prediction = model.predict([features["user_age"], features["last_login_days"]])
except redis.ConnectionError:
    print("\nRedis is not reachable on localhost:6379 — skipping the cache demo.")
    print("Start one with:  docker run -d -p 6379:6379 redis:7-alpine")
```
- This is the **realistic use case** for MessagePack in an ML system: caching feature vectors in Redis so a serving layer doesn't recompute them on every request.
- **Gracefully degrades**: if Redis isn't running, the script prints a notice and exits cleanly rather than crashing — the JSON-vs-MessagePack comparison above still runs fine on its own.
- To actually run this half: `docker run -d -p 6379:6379 redis:7-alpine`

### Installing what this example needs
This file's dependencies (`msgpack`, `redis`) are **not** part of the core API — they're an opt-in extra:
```bash
pip install -e ".[msgpack]"
python msgpack_example.py
```

## Format comparison (general knowledge, extending the demo above)

| Format | Type | Human readable | Best for |
|---|---|---|---|
| **JSON** | Text | ✅ Yes | REST APIs, configs, logging |
| **MessagePack** | Binary | ❌ No | Caching feature vectors, high-throughput calls |
| **Pickle** | Binary | ❌ No | Python-only model files — **never expose via an API** |

> [!danger] Never deserialize Pickle from an untrusted source
> A malicious pickle file can execute arbitrary code on load. Fine for saving your own model file locally; never accept one over a network request.

---

## ONNX (Open Neural Network Exchange)

### What is it, precisely
An open, framework-neutral **file format for ML models**. Train in whatever framework you like (PyTorch, TensorFlow, scikit-learn via `skl2onnx`), export once to `.onnx`, then run it anywhere with an ONNX **runtime** — C++, Python, a phone, the browser, specialized accelerators — with **no PyTorch installed on the serving side**.

For MLOps this decouples the *training* stack from the *serving* stack: the heavyweight framework stays in training; production ships a small, portable artifact.

### What actually gets packed into an `.onnx` file
An `.onnx` file is a **protobuf** containing three things:

| Part | What it is | In this project's export |
|---|---|---|
| **The graph** | The forward pass flattened into a static directed acyclic graph of nodes — not your Python code, just the math it performs. Python control flow, classes, and method structure are gone; the exporter *traces* the forward pass and records the operations it saw | one `Gemm` node (matrix multiply + bias) |
| **The operators (ops)** | Each node is one operator from ONNX's standardized catalog (~190 ops: `Conv`, `MatMul`, `Relu`, `Softmax`...), versioned by **opset** number. Any runtime supporting opset 18 can execute any graph using only opset-18 ops | `Gemm` from the standard opset |
| **The weights (initializers)** | Every learned parameter, baked into the file as raw tensors — this is why an `.onnx` file is self-contained | the linear layer's weight `[[2.0, 0.5]]` and bias |

> [!warning] What is NOT converted
> Training logic (loss, optimizer, gradients), your Python classes, and any preprocessing **outside** the model's `forward()`. If a scaler/encoder isn't part of the exported graph, the serving side must reproduce it exactly — this is the same "scaler left behind" trap shown in the bad notebook (see [[01 - MLOps Maturity Model]]).

### The actual export script (`pytorch_to_onnx.py`), in full
```python
import torch
import torch.nn as nn
import torch.onnx


class RideDurationTorchModel(nn.Module):
    """PyTorch version of the ride-duration heuristic.

    A single linear layer over [distance_km, passengers]. Weights are
    initialized to reproduce src.model.RideDurationModel:
    duration = distance / 0.5 + passengers * 0.5 = 2*distance + 0.5*passengers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[2.0, 0.5]]))
            self.linear.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


model = RideDurationTorchModel()
model.eval()   # disable dropout/batchnorm training mode

dummy_input = torch.randn(1, 2)   # [distance_km, passengers] — only the SHAPE matters

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,       # bundle trained weights into the file
    opset_version=17,         # ONNX operator set version
    input_names=["features"],
    output_names=["duration"],
    dynamic_axes={             # allow variable batch size at inference
        "features": {0: "batch_size"},
        "duration": {0: "batch_size"},
    },
)

import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)   # raises if graph is invalid
print("Inputs: ", [i.name for i in onnx_model.graph.input])
print("Outputs:", [o.name for o in onnx_model.graph.output])
```

| Detail | Why it matters |
|---|---|
| `model.eval()` | Forgetting this leaves dropout/batchnorm active during export → silently wrong predictions later |
| `dummy_input` | Only shape and dtype matter — PyTorch traces the graph by literally running the dummy through it once |
| `dynamic_axes` | Without it, the model is locked to batch size 1 forever |
| `onnx.checker.check_model()` | A broken export can fail silently in some runtimes — always validate before shipping |

### Running it
```bash
pip install -e ".[onnx]"     # torch, onnx, onnxscript — kept out of core deps, they're heavy
python pytorch_to_onnx.py
```
Writes `model.onnx` to the project root and prints:
```
Inputs:  ['features']
Outputs: ['duration']
```
> A couple of **non-fatal warnings are expected**: the opset gets auto-bumped **17→18** (you asked for 17, the exporter used 18 — this is normal, not an error), and a `torchvision not installed` notice. Neither affects the exported model.

### ⚠️ The catch: new/exotic models may need ops ONNX doesn't have
Because every graph node must map to a standardized op, a model is only exportable if **every operation in its forward pass has an ONNX equivalent**. Established architectures (CNNs, ResNets, standard transformers) export cleanly — cutting-edge or custom models often don't (novel attention variants, custom CUDA kernels, exotic scatter/gather patterns). The failure looks like:
```
UnsupportedOperatorError: Exporting the operator 'aten::my_fancy_op' to ONNX
opset version 18 is not supported
```

**Escalation path, cheapest first:**
1. **Try a newer opset** — the op may have been added in a later spec revision
2. **Rewrite as a composition of supported ops** — e.g. approximate a custom activation with standard `Sigmoid`/`Mul`/`Add` — cheap when possible, but changes the graph and sometimes the numerics
3. **Register a custom translation** — teach the exporter your op via a custom symbolic function (PyTorch has an [official tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html) for exactly this)
4. **Implement the op in the runtime yourself** — ONNX Runtime custom op (C++/Python) or the `com.microsoft` contrib-op domain. Full control, but you now maintain custom serving code — much of the "portable artifact" benefit is gone

> [!tip] The practical takeaway
> **ONNX exportability is a deployment constraint to check *before* committing to an architecture**, not an afterthought. A 30-second `torch.onnx.export` smoke test on a prototype (exactly what this script does) tells you whether your serving plan works before you've spent weeks training.

### What converts, and how well (general knowledge)

| Source framework | Converter | Notes |
|---|---|---|
| PyTorch | `torch.onnx` (built-in) | Best supported — native export API |
| scikit-learn | `skl2onnx` | Covers most estimators + Pipelines |
| TensorFlow / Keras | `tf2onnx` | Some custom layers need manual ops |

### When to use ONNX, when not to

| ✅ Use ONNX when | ❌ Skip ONNX when |
|---|---|
| Serving from C++, Java, C#, or the browser | You're serving from Python anyway |
| Dropping a training framework from your image | Model uses custom ops the converter can't handle |
| Need TensorRT / OpenVINO / CoreML acceleration | Model changes weekly — one more export step to maintain |
