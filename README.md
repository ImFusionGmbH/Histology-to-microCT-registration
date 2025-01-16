# ML-based Histology to Micro-CT Registration
This repository contains a reference open source implementation of the paper [2D-3D Deformable Image Registration of Histology Slide and Micro-CT with ML-based Initialization (Chen et al. 2024)](https://arxiv.org/abs/2410.14343) as an ImFusionSuite plugin.
The feature maps of histology image and micro-CT are computed using the DISA 2D model. 
The 2D-3D registration is then initialized using a global feature map registration.
The 2D-3D registration is later refined by optimizing the sampling plane parameters using 2D-2D intensity-based image registration.


**Requires ImFusion SDK (with ImFusion HistologyPlugin) version 3.12.0 or newer**

**NOTE** The algorithm is also part of the ImFusion SDK.