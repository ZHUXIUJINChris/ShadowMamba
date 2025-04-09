# ShadowMamba
ShadowMamba: State-Space Model with  Boundary-Region Selective Scan for Shadow  Removal

Abstract: Image shadow removal is a common low-level vision problem. Shadows cause sudden brightness changes in some areas, which can affect the accuracy of downstream tasks. Currently, Transformer-based shadow removal methods improve computational efficiency by using a window mechanism. However, this approach reduces the effective receptive field and weakens the ability to model long-range dependencies in shadow images. Recently, Mamba has achieved significant success in computer vision by modeling long-sequence information globally with linear complexity. However, when applied to shadow removal, its original scanning mechanism overlooks the semantic continuity along shadow boundaries, and the coherence within each region. To solve this issue, we propose a new boundary-region selective scanning mechanism that scans shadow, boundary, and non-shadow regions separately, making pixels of the same type closer in the sequence. This increases semantic continuity and helps the model understand local details better. Incorporating this idea, we design the first Mamba-based lightweight shadow removal model, called ShadowMamba. It uses a hierarchical combination U-Net structure, which effectively reduces the number of parameters and computational complexity. Shallow layers rely on our boundary-region selective scanning to capture local details, while deeper layers use global cross-scanning to learn global brightness features. Extensive experiments show that ShadowMamba outperforms current state-of-the-art models on ISTD+, ISTD, and SRD datasets, and it also requires fewer parameters and less computational cost. (Code will be made available upon paper acceptance.)

![ShadowMamba](https://github.com/user-attachments/assets/f15f9740-2368-4fb4-aa5c-9d9612a9701a)
