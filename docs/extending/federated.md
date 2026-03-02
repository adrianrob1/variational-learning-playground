# Extending the Federated Benchmark

This guide explains how to add new components specific to the federated learning pipeline.

---

## 1. Adding a New Partitioned Dataset

The federated benchmark uses the `PartitionedDataset` wrapper to split standard datasets among clients. 

### Implementation
If you are adding a completely new base dataset (e.g., SVHN), you create a loader function in [src/vldatasets/partitioned](../../src/vldatasets/partitioned/).

```python
from .core import PartitionedDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_partitioned_mydata_loaders(
    data_dir: str,
    num_clients: int,
    alpha1: float = 1e6,
    alpha2: float = 0.5,
    seed: int = 42,
    batch_size: int = 32,
    # ... other args
):
    # Load base dataset
    train_dataset = MyCustomDataset(root=data_dir, train=True, download=True)
    test_dataset = MyCustomDataset(root=data_dir, train=False, download=True)
    
    # Wrap in PartitionedDataset
    partitioned_train = PartitionedDataset(
        train_dataset,
        num_clients,
        num_classes=10, # Adjust for your dataset
        alpha1=alpha1,
        alpha2=alpha2,
        seed=seed,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return partitioned_train, test_loader
```

### Integration
### Integration
Create a Hydra config at `conf/federated/dataset/mydata.yaml`, using a `loader` block with `_target_` to hook up your new loader function:
```yaml
name: "mydata"
num_classes: 10
in_channels: 3
img_size: 32
input_dim: 3072
batch_size: 64
workers: 2
loader:
  _target_: vldatasets.partitioned.mydata.get_partitioned_mydata_loaders
  _partial_: true
```

> [!TIP]
> The federated execution scripts (`train_federated.py`, `track_elbo.py`) utilize `hydra.utils.instantiate(cfg.dataset.loader)` to dynamically load your dataset. This means you do **not** need to modify the internal data loading dispatch logic in those scripts; your custom dataset configuration plugs in automatically!

---

## 2. Adding a New Federated Method (Aggregator/Optimizer combination)

In this framework, a federated "Method" usually involves two pieces:
1. The local client optimizer.
2. The server-side orchestration strategy (e.g., `FedAvg`, `FedProx`, `FedDyn`).

### Implementation
If your method introduces a new local optimizer, implement it in [src/vloptimizers/federated](../../src/vloptimizers/federated/). 

If your method requires new distributed orchestration logic (like sending dual variables or computing custom aggregated posteriors), you need to update:
- `FederatedWorker._compute_loss` (for local regularization terms like Proximal or Dynamic).
- `FederatedWorker.update_dual` (if clients maintain dual states).
- `FederatedOrchestrator._distribute` and `FederatedOrchestrator._aggregate` in [src/vlbench/federated](../../src/vlbench/federated/).

### Integration
Create a Hydra config at `conf/federated/method/mymethod.yaml` to leverage Hydra instantiation for the local optimizer. We use `_partial_: true` because the parameters are passed dynamically during the federated training loop:

```yaml
name: "mymethod"
_target_: vloptimizers.federated.mymethod_optimizer.MyOptimizer
_partial_: true
lr: 0.1
# Added hyperparams that FederatedWorker/FederatedOrchestrator might read:
rho: 0.01 
mu: 0.01
```

> [!TIP]
> The federated execution scripts (`train_federated.py`, `track_elbo.py`) use Python's `inspect.signature` to automatically detect if your custom optimizer requires additional parameters (like `ess`, `prior_mean`, `dual_mean`, etc.) and seamlessly injects them without any hardcoded `if` statements!

---

## 3. Adding a New Federated Model

Federated tasks often use specialized light-weight models because testing involves many clients.

### Implementation
Add your model architecture to [src/vlbench/federated/models.py](../../src/vlbench/federated/models.py). 

```python
class MyFedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # ...
        
    def forward(self, x):
        # ...
```

### Integration
Because the federated scripts (`train_federated.py`, `track_elbo.py`, etc.) leverage `hydra.utils.instantiate()`, you simply need to create a configuration file for your model in `conf/federated/model/`:

**`conf/federated/model/myfedmodel.yaml`**
```yaml
name: "myfedmodel"
_target_: mypackage.models.MyFedModel
# We can dynamically pull dimensions from the dataset config!
input_dim: ${dataset.input_dim}
output_dim: ${dataset.num_classes}
```

Then run scripts with `model=myfedmodel`.
