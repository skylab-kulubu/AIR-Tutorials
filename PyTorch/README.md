# Pytorch Tutorial

PyTorch is an open source Deep Learning library based on Torch library. It is primarily developed by Facebook's AI Research lab (FAIR).
PyTorch provides two high-level features:

- Tensor computing (like NumPy) with strong acceleration via graphics processing units (GPU)
- Deep neural networks built on a tape-based autodiff system

## About Repository
- This repository contains my PyTorch works, projects.
- This repository introduces PyTorch modules, neural networks on PyTorch.

## Modules  

- Autograd module:
PyTorch uses a method called automatic differentiation. A recorder records what operations have performed, and then it replays it backward to compute the gradients. This method is especially powerful when building neural networks to save time on one epoch by calculating differentiation of the parameters at the forward pass.

- Optim module
torch.optim is a module that implements various optimization algorithms used for building neural networks. Most of the commonly used methods are already supported, so there is no need to build them from scratch.

- nn module
PyTorch autograd makes it easy to define computational graphs and take gradients, but raw autograd can be a bit too low-level for defining complex neural networks. This is where the nn module can help.
