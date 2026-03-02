# Extending the Benchmark

This guide explains how to add new components (datasets, methods, and models) to the `vlbench` framework and integrate them into the existing benchmarks.

---

## 1. Adding a New Dataset

To add a new dataset, you implement the data loading logic and create a Hydra configuration. No manual registry updates are required.

### Implementation
Add your loader functions to [dataloaders.py](../src/vlbench/datasets/dataloaders.py):

1. **Implement Loader Functions**:
   ```python
   def get_mydata_train_loaders(data_dir, train_val_split, workers, pin_memory, tbatch, vbatch):
       # Return (train_loader, val_loader)
       ...

   def get_mydata_test_loader(data_dir, workers, pin_memory, batch):
       # Return test_loader
       ...
   ```

### Integration
Create a Hydra configuration file:
- **Path**: `conf/indomain/dataset/mydata.yaml`
- **Content**:
  ```yaml
  _target_: vlbench.datasets.DatasetConfig
  name: mydata
  outclass: 10
  ntrain: 50000
  ntest: 10000
  insize: 32
  train_loader_f: 
    _target_: vlbench.datasets.dataloaders.get_mydata_train_loaders
    _partial_: true
  test_loader_f:
    _target_: vlbench.datasets.dataloaders.get_mydata_test_loader
    _partial_: true
  ```

---

## 2. Adding a New Method (Optimizer)

Methods correspond to optimizers and their associated training logic (e.g., MC sampling).

### Implementation
1. **Create Optimizer**: Add your optimizer class to [src/vloptimizers/](src/vloptimizers/).
2. **Handle Special Logic**: If your method requires custom updates (like IVON's sampled parameters), update the `do_trainbatch_*` functions in [train.py](../src/vlbench/indomain/train.py) and the dispatch logic in `main`.

### Integration
Create a Hydra configuration file:
- **Path**: `conf/indomain/method/mymethod.yaml`
- **Content**:
  ```yaml
  _target_: vloptimizers.MyOptimizer
  name: mymethod
  lr: 0.1
  # Add other hyperparameters
  ```

---

## 3. Adding a New Model

### Implementation
Add your architecture to [src/vlbench/models/_registry.py](../src/vlbench/models/_registry.py):

1. **Define Factory Function**:
   ```python
   def mymodel(outclass: int, input_size: int = 32, **kwargs) -> torch.nn.Module:
       return MyModel(num_classes=outclass, input_size=input_size, **kwargs)
   ```
   > [!TIP]
   > It is highly recommended to support `input_size` as an argument. The benchmark automatically passes the `insize` defined in the dataset configuration to all models during instantiation.

2. **Register Model**:
   Add it to the `STANDARDMODELS` dictionary.

### Integration
Create a Hydra configuration file:
- **Path**: `conf/indomain/model/mymodel.yaml`
- **Content**:
  ```yaml
  name: mymodel
  ```

---

## 4. Using Hugging Face Datasets

You can leverage the `datasets` library to load datasets from Hugging Face Hub. A generic loader is provided in `dataloaders.py`.

### Integration
Create a Hydra configuration using `get_hf_train_loaders` and `get_hf_test_loader`:

- **Path**: `conf/indomain/dataset/mnist_hf.yaml`
- **Content**:
  ```yaml
  _target_: vlbench.datasets.DatasetConfig
  name: mnist_hf
  outclass: 10
  ntrain: 60000
  ntest: 10000
  insize: 28
  train_loader_f: 
    _target_: vlbench.datasets.dataloaders.get_hf_train_loaders
    path: mnist
    image_col: image # Configurable column names
    label_col: label
    _partial_: true
  test_loader_f:
    _target_: vlbench.datasets.dataloaders.get_hf_test_loader
    path: mnist
    image_col: image
    label_col: label
    _partial_: true
  ```

---

## 5. Running the Benchmark

Once registered, you can use your new components via Hydra overrides:

```bash
uv run python -m vlbench.indomain.train \
  dataset=mydata \
  model=mymodel \
  method=mymethod \
  method.lr=0.01
```

### Verification
Always verify your integration with a short run:
```bash
uv run python -m vlbench.indomain.train ... epochs=1 warmup=0
```
