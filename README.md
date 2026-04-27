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
