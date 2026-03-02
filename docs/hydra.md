# Hydra Configuration Guide

The `vlbench` framework uses [Hydra](https://hydra.cc/) to manage complex configurations. This guide explains how Hydra works in practice and how the syntax affects the execution of your benchmarks.

## 1. Directory Structure

Configurations are organized into **Config Groups**:

```text
conf/indomain/
├── config.yaml       # The "Main" configuration (Entry point)
├── dataset/          # Config Group: Dataset settings
├── method/           # Config Group: Optimization methods
└── model/            # Config Group: Model architectures
```

---

## 2. Core Concepts & Syntax

### Composition (The Defaults List)

**In Practice**: Composition determines the "starting state" of your experiment. Instead of one giant 500-line file, we piece together smaller blocks.

**Syntax**:
- `- group: name`: Loads the file `conf/indomain/group/name.yaml`.
- `_self_`: Tells Hydra where to apply settings from the current file relative to the included ones.

```yaml
# conf/indomain/config.yaml
defaults:
  - method: sgd       # Start with SGD defaults
  - dataset: cifar10  # Add CIFAR-10 data settings
  - model: resnet20   # Add ResNet-20 architecture
  - _self_            # Apply overrides from THIS file last
```

---

### Command-Line Overrides

**In Practice**: Overrides allow you to "live-patch" the configuration at runtime without touching any YAML files. This is the primary way to iterate on experiments.

**Syntax**:
- **Dot Notation (`path.to.key=value`)**: Replaces a specific value deep in the config tree.
- **Assignment (`group=name`)**: Swaps out an entire sub-config file for another.

| Action                | Command Syntax     | What it does                                                      |
| :-------------------- | :----------------- | :---------------------------------------------------------------- |
| **Swap Method**       | `method=ivon`      | Discards `sgd.yaml` settings and loads `ivon.yaml` instead.       |
| **Change Hyperparam** | `method.lr=0.01`   | Only changes the `lr` value inside the currently loaded `method`. |
| **Global Param**      | `epochs=50`        | Changes a top-level parameter in the main `config.yaml`.          |
| **Add Param**         | `+method.beta=0.9` | Adds a new parameter `beta` that wasn't in the original config.   |
| **Remove Param**      | `~method.momentum` | Removes the parameter `momentum` from the configuration.          |

---

### Multirun (Sweeps)

**In Practice**: Multirun automates "looping" over experiments. It generates multiple distinct jobs from a single command, each with its own subdirectory and logs.

**Syntax**:
- **The Flag**: `-m` or `--multirun`.
- **Comma-Separated**: `key=val1,val2`.
- **Range (Optional)**: `key=range(start,end,step)`.

**Example**:
```bash
# Trains 3 models (ResNet-20, ResNet-18Wide, DenseNet) across 2 methods (SGD, IVON)
# Total: 3 * 2 = 6 independent training runs.
uv run python -m vlbench.indomain.train -m model=resnet20,resnet18wide,densenet121 method=sgd,ivon
```

---

### Instantiation and Targets

**In Practice**: This is where "Config turns into Code". Instead of hardcoding `if opt == "ivon": IVON(...)`, the code simply asks Hydra to "create whatever the config says to create".

**Syntax**:
- `_target_`: The full Python path to the class or function.
- `_partial_` (optional): If `true`, returns a callable function with parameters pre-filled, rather than an active object.

#### Standard Instantiation
**Config (`method/ivon.yaml`)**:
```yaml
_target_: vloptimizers.ivon.IVON
lr: 0.1
ess: 300000.0
```

**Python Implementation**:
```python
# Hydra imports the class, passes lr/ess, and returns a live IVON instance.
optimizer = hydra.utils.instantiate(cfg.method, params=model.parameters())
```

#### Partial Instantiation (`_partial_: true`)
Used when you want a "lazy" constructor or a callback. Instead of running the function immediately, Hydra returns a function that can be called later with additional arguments.

**Config (`metric/bleu.yaml`)**:
```yaml
_target_: vlbench.text_generation.metrics.bleu
_partial_: true
```

**Python Implementation**:
```python
# Returns the 'bleu' function itself (or a wrapper with pre-filled args)
metric_fn = hydra.utils.instantiate(cfg.metric)

# Can be called later with data
scores, mean = metric_fn(hypotheses, references)
```

---

### Variable Interpolation

**In Practice**: Interpolation enforces consistency between settings. If you change a model name, the `save_dir` updates automatically to match, preventing you from accidentally overwriting old results.

**Syntax**: `${path.to.variable}`

```yaml
# Inside config.yaml
save_dir: outputs/${dataset.name}/${method.name}/${seed}
```

---

## 3. Practical Workflow Summary

1.  **Define Default**: Set your "baseline" in `config.yaml` using `defaults`.
2.  **Trial Run**: Test a small change via override: `method.lr=0.001 epochs=1`.
3.  **The Sweep**: Once satisfied, launch a full benchmark: `-m method=ivon,sgd seed=0,1,2,3,4`.

---

## References
For deeper technical details, see the [Official Hydra Docs](https://hydra.cc/docs/intro/).
