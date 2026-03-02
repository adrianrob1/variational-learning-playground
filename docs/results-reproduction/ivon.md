# Reproducing Results from `ivon-experiments`

This document details how to reproduce the experiments from the [IVON (Improved Variational Online Newton)](https://github.com/team-approx-bayes/ivon) paper using the `variational-learning-benchmark`. 

The benchmark integrates the `ivon` optimizer and the necessary models (e.g., `resnet20`, `preresnet110`, `densenet121`, `resnet18wide`) to run the experiments reported in the paper.

---

## 1. In-Domain Benchmark (CIFAR-10)

This covers the in-domain Bayesian deep learning experiments on CIFAR-10.
The training script will automatically evaluate the model on the test set.

### IVON

To train and test the standard IVON model with 5 seeds (0-4):

```bash
uv run python -m vlbench.indomain.train method=ivon dataset=cifar10 model=resnet20 seed=0
uv run python -m vlbench.indomain.train method=ivon dataset=cifar10 model=resnet20 seed=1
uv run python -m vlbench.indomain.train method=ivon dataset=cifar10 model=resnet20 seed=2
uv run python -m vlbench.indomain.train method=ivon dataset=cifar10 model=resnet20 seed=3
uv run python -m vlbench.indomain.train method=ivon dataset=cifar10 model=resnet20 seed=4
```

### Multi-IVON

Multi-IVON is an ensemble of multiple independently trained IVON instances (similar to Deep Ensembles). You simply run training with different seeds and ensemble their predictions during evaluation.

```bash
# Run multiple instances with different seeds (e.g., seeds 5-24)
uv run python -m vlbench.indomain.train method=ivon dataset=cifar10 model=resnet20 seed=5
# ... repeat for other seeds
```

---

## 2. Image Classification (Single GPU)

This section covers the Image Classification ablation studies in the paper. Different combinations of datasets and models can be evaluated.

Supported Datasets: `cifar10`, `cifar100`, `tinyimagenet`
Supported Models: `resnet20`, `resnet18wide`, `densenet121`, `preresnet110`

### Training IVON

```bash
uv run python -m vlbench.image_classification.train \
    resnet20 \
    cifar10 \
    -opt ivon \
    -s 0
```

*(Swap the dataset and architecture to reproduce other runs, and vary `-s` from 0-4).*

---

## 3. Out-Of-Distribution (OOD) Benchmark

This covers the OOD experiments on SVHN and Flowers102 using models pre-trained on CIFAR-10. Ensure you have run the in-domain IVON training on CIFAR-10 first.

### Evaluate on SVHN

```bash
uv run python -m vlbench.ood.run runs/cifar10/ivon/resnet20/0 --ood_dataset svhn -tr 64
```

### Evaluate on Flowers102

```bash
uv run python -m vlbench.ood.run runs/cifar10/ivon/resnet20/0 --ood_dataset flowers102 -tr 64
```

*(The `-tr 64` specifies the number of MC posterior samples to use during inference).*

---

## 4. Multi MC Sample Ablation

To evaluate the impact of the number of Monte Carlo (MC) samples during test time, you can use the `eval_mcsamples.py` script.

Assuming you have a trained model at `runs/cifar10/ivon/resnet20/0`:

```bash
uv run python scripts/eval_mcsamples.py save_dir=runs/cifar10/ivon/resnet20/0 mc_samples_list="1,2,4,8,16,32,64"
```

---

## 5. BDL Competition datasets

For the NeurIPS 2021 Bayesian Deep Learning competition datasets (CIFAR-10, MedMNIST, UCI).

**CIFAR-10** (using AlexNet):
```bash
uv run python -m vlbench.bdl_competition.train method=ivon dataset=cifar10
```

**MedMNIST** (using CNN):
```bash
uv run python -m vlbench.bdl_competition.train method=ivon dataset=medmnist
```

**UCI** (using MLP):
```bash
uv run python -m vlbench.bdl_competition.train method=ivon dataset=uci
```

---

## 6. ImageNet

For the ImageNet distributed training experiments, use PyTorch's distributed launcher and ensure `ffcv` is installed if you want fast data loading.

```bash
uv run torchrun --nproc_per_node=8 \
    src/vlbench/image_classification/distributed_train.py \
    method=ivon_imagenet \
    dataset=imagenet \
    model=resnet50_imagenet
```

> [!NOTE]
> Training IVON on ImageNet requires multiple GPUs to complete in a reasonable timeframe. Adjust `--nproc_per_node` depending on the number of GPUs available.

