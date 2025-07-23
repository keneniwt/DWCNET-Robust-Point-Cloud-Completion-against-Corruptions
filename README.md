# Denoising-While-Completing-Network(DWCNET):Robust-Point-Cloud-Completion-against-Corruptions 

### This repository contains PyTorch implementation for Denoising-While-Completing-Network(DWCNET):Robust-Point-Cloud-Completion-against-Corruptions.[ArXiv](https://arxiv.org/abs/2507.16743)
Point cloud completion is crucial for 3D computer vision tasks in autonomous driving, augmented reality, and robotics. However, obtaining clean and complete point clouds from real-world environments is challenging due to noise and occlusions. Consequently, most existing completion networks—trained on synthetic data—struggle with real-world degradations. In this work, we tackle the problem of completing and denoising highly corrupted partial point clouds affected by multiple simultaneous degradations. To benchmark robustness, we introduce the Corrupted Point Cloud Completion Dataset (CPCCD), which highlights the limitations of current methods under diverse corruptions. Building on these insights, we propose DWCNet (Denoising-While-Completing Network), a completion framework enhanced with a Noise Management Module (NMM) that leverages contrastive learning and self-attention to suppress noise and model structural relationships. DWCNet achieves state-of-the-art performance on both corrupted synthetic and real-world datasets. 

## Dataset

Our Corrupted-Point-Cloud-Completion-Dataset(CPCCD) can be downloaded [here](https://zenodo.org/records/16085700?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhkZTc1OTc4LWMwZWUtNDAxYS1hNDk5LTY2MjIzZWUyZWMzMSIsImRhdGEiOnt9LCJyYW5kb20iOiI1NTQ5NjQzYTFiODhlODg2ZmM0NjZkZDAzNmNiM2IxOCJ9.fdDDCKOgizn-FFAl80c_PCAOCoggpIgNwMdyrMATihoYhZGlI6CgWSi8GRkmIIpoWKhnDVPc5IPuViq6EbMKZQ). Our datasets is based on the PCN completion benchmark dataset which can be downloaded [here](https://gateway.infinitescript.com/s/ShapeNetCompletion).  

## Code
##### Coming Soon!

## Licence
MIT Licence

## Citation
If your find our work useful, please consider citing: 

@misc{tesema2025denoisingwhilecompletingnetworkdwcnetrobust,
      title={Denoising-While-Completing Network (DWCNet): Robust Point Cloud Completion Under Corruption}, 
      author={Keneni W. Tesema and Lyndon Hill and Mark W. Jones and Gary K. L. Tam},
      year={2025},
      eprint={2507.16743},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.16743}, 
}






