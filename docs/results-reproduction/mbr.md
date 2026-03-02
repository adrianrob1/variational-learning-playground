# Reproducing ICLR 2025 MBR Uncertainty Experiments

This guide explains how to use the `variational-learning-benchmark` to reproduce the Minimum Bayes Risk (MBR) decoding experiments for text generation, as presented in the ICLR 2025 paper: *Uncertainty-Aware Minimum Bayes Risk Decoding*.

**Note:** In this repository, we strictly provide tools to reproduce the Hugging Face (Gemma) track. For reproducing the specific `fairseq` track (IWSLT14), please refer to the original paper's repository, as porting its modified `fairseq` fork is highly intrusive.

## 1. Overview

The implemented components are located under `src/vlbench/text_generation/`. It consists of:
- **`train.py`:** fine-tunes a causal or seq2seq language model with our native `vloptimizers.ivon.IVON` optimizer, properly managing its pre-sampling and Hessian state.
- **`predict.py`:** runs hypothesis generation via the HuggingFace `Trainer` and native generation algorithms, applying MC dropout or IVON sampling to produce diverse candidate hypotheses.
- **`mbr.py`:** implements the MBR decoding logic (`mbr_corpus`) that ranks and selects the best hypotheses based on expected utility.
- **`metrics.py`:** wraps the evaluation metrics like `BLEU` and `BLEURT`. *(Note: `BERTScore` and `COMET` were excluded natively due to upstream dependency/caching incompatibilities but can be run independently using the raw generated texts).*

## 2. Using Hydra for Execution

Instead of using custom `.json` training configurations as in the original repository, you should configure training via standard Hydra configurations (or compose them dynamically). A typical run uses the `test.py` or `train.py` wrappers by instantiating the components.

### Training

To train a model utilizing IVON for uncertainty quantification:

```yaml
# config/method/mbr_ivon.yaml
optimizer:
  _target_: vloptimizers.ivon.IVON
  lr: 2e-4
  mc_samples: 1
  weight_decay: 1e-4

model:
  type: causal_lm
  model_name_or_path: "google/gemma-2b"

training_args:
  per_device_train_batch_size: 4
  num_train_epochs: 3
```

Execute your custom script or use Python to pass the instantiated Hydra configuration directly to the `train` routine:

```python
from vlbench.text_generation.train import train
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    train(cfg)
```

### Inference & MBR Evaluation

Generate samples using your test configurations:

```python
from vlbench.text_generation.predict import predict

# Ensure 'ivon_inference', 'mc_samples', and 'generation' are in cfg
predictions = predict(cfg)
```

Then, evaluate and pick the best hypotheses:

```python
from vlbench.text_generation.mbr import mbr_corpus, select_best_hypotheses
from vlbench.text_generation.metrics import bleu

# hyps: (num_samples, num_hypotheses)
# refs: usually the same as hyps for pure MBR
utilities = mbr_corpus(hyps, metric=bleu)
best_predictions = select_best_hypotheses(hyps, utilities)
```

## 3. Environment Constraints

If you intend to use `BLEURT` during MBR selection natively, you must manually install the `bleurt` library from its original GitHub repository prior to execution, as it is not distributed via standard package indices.
