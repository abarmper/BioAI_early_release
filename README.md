# BioAI

BioAI is a 3D medical image segmentation framework for research on expert-augmented convolutional networks. The central contribution is **Soft-MoE MedNeXt**, which extends the MedNeXt convolutional encoder–decoder with a differentiable Soft Mixture of Experts layer at the bottleneck, enabling conditional expert specialization without self-attention or its quadratic cost over dense 3D voxel grids.

The framework wraps this architecture in a fully automated pipeline following nnUNet v2 conventions: dataset fingerprinting, topology-aware model construction, fixed-iteration training, sliding-window ensemble inference, and optional post-processing — all driven by a composable Hydra configuration system.

## Key features

- **Soft-MoE bottleneck**: differentiable soft routing of spatial tokens across multiple expert FFNs; configurable number of experts, placement mode (conservative / full), and optional axial positional embeddings.
- **Multi-architecture support**: MedNeXt, nnMedNeXt, MedMoENeXt, nnMedMoENeXt, MONAI UNet / DynUNet / SwinUNETR / UNETR / SegResNet / AttentionUNet, custom nnUNet (plain + residual encoder), NNFormer.
- **Automated planning**: voxel spacing, patch size, network depth, and pooling strides computed from the dataset fingerprint; isotropic and anisotropic configurations supported.
- **nnUNet v2 training loop**: mixed precision, `torch.compile`, PolyLR scheduling, EMA early stopping, foreground oversampling, deep supervision.
- **Ensemble inference**: per-fold checkpoint loading with CPU logit averaging and optional mirror test-time augmentation.
- **Experiment tracking**: Weights & Biases integration with per-fold and cross-validation aggregation.
- **Pretraining / fine-tuning**: UpKern spatial weight interpolation, transfer learning from external checkpoints, layer freezing with gradual unfreezing and per-group learning rate scaling.

## Requirements

BioAI is implemented in Python and relies on a CUDA-enabled PyTorch stack for GPU-accelerated 3D medical image segmentation. The provided environment includes, among others:

- `torch==2.11.0+cu126` and `torchvision==0.26.0+cu126`
- CUDA 12.6-related packages, including `cuda-toolkit==12.6.3`
- `monai==1.5.2`
- `hydra-core==1.3.2` and `omegaconf==2.3.0`
- `numpy==2.4.3`, `scipy==1.17.1`, `pandas==3.0.2`, and `scikit-image==0.26.0`
- `nibabel==5.4.2` and `simpleitk==2.5.3` for medical image I/O
- `einops==0.8.2`, `timm==1.0.26`, and `triton==3.6.0`
- `wandb==0.26.0` for experiment tracking

A complete list of pinned dependencies is provided in `requirements.txt`.
