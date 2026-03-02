# Tasks Index

The `variational-learning-benchmark` supports several evaluation tasks, each with its own setup and scripts. Below is an index of all supported tasks and their corresponding guides:

- [In-Domain Training and Evaluation](in_domain.md): Standard Training/Eval (CIFAR, TinyImageNet) 
- [Out-Of-Distribution Evaluation](ood.md): Out-of-Distribution (SVHN, Flowers102) using In-Domain Checkpoints
- [Robustness Evaluation](robustness.md): Data Corruptions (CIFAR-C) using In-Domain Checkpoints
- [NeurIPS BDL Competition Tasks](bdl_competition.md): BDL Competition Tasks (Diabetic Retinopathy, MedMNIST, etc.)
- [Distributed ImageNet Training](imagenet.md): Large-scale Distributed Training using FFCV and ImageNet
- [MC Ablation Studies](mc_ablation.md): Monte Carlo Sample counts study
- [Federated Learning](federated.md): Federated Learning across clients
- [Text Generation (MBR)](../results-reproduction/mbr.md): Text Generation & MBR Decoding


For general information on how to configure these tasks using Hydra, please refer to the [Hydra Configuration Guide](../hydra.md).
