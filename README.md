# Denoising-While-Completing-Network(DWCNET):Robust-Point-Cloud-Completion-against-Corruptions 
By [Keneni Worku Tesema](https://scholar.google.com/citations?user=7ryP6-kAAAAJ&hl=en&oi=ao), [Lyndon Hill](https://ieeexplore.ieee.org/author/420404236342407), [Mark W. Jones](https://scholar.google.co.uk/citations?hl=en&user=zlThfKsAAAAJ&view_op=list_works&sortby=pubdate), [Gary Tam](https://scholar.google.co.uk/citations?user=MMhCPiwAAAAJ&hl=en)

### This repository contains PyTorch implementation for Denoising-While-Completing-Network(DWCNET):Robust-Point-Cloud-Completion-against-Corruptions 
Accepted for Computers and Graphics and EG Symposium on 3D Object Retrieval 2025 (3DOR'25)

#### [[ArXiv]](https://arxiv.org/abs/2507.16743) [[SSRN]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5360323) [[Dataset]](https://zenodo.org/records/16085700)
Point cloud completion is crucial for 3D computer vision tasks in autonomous driving, augmented reality, and robotics. However, obtaining clean and complete point clouds from real-world environments is challenging due to noise and occlusions. Consequently, most existing completion networks—trained on synthetic data—struggle with real-world degradations. In this work, we tackle the problem of completing and denoising highly corrupted partial point clouds affected by multiple simultaneous degradations. To benchmark robustness, we introduce the Corrupted Point Cloud Completion Dataset (CPCCD), which highlights the limitations of current methods under diverse corruptions. Building on these insights, we propose DWCNet (Denoising-While-Completing Network), a completion framework enhanced with a Noise Management Module (NMM) that leverages contrastive learning and self-attention to suppress noise and model structural relationships. DWCNet achieves state-of-the-art performance on both corrupted synthetic and real-world datasets. 

<img width="4009" height="2977" alt="completion results before finetuning" src="https://github.com/user-attachments/assets/81194ecf-76fa-4299-a80c-72bf8c1e1bfd" />
<img width="4169" height="3084" alt="completion results after finetuning" src="https://github.com/user-attachments/assets/7f89f21e-0044-45fc-8383-51eb51c0e696" />


## Dataset

Our Corrupted-Point-Cloud-Completion-Dataset(CPCCD) can be downloaded [Zenodo](https://zenodo.org/records/16085700). Our datasets is based on the PCN completion benchmark dataset which can be downloaded [here](https://gateway.infinitescript.com/s/ShapeNetCompletion).  

## PreTrained Models
##### Coming Soon!

## Licence
MIT Licence

## Citation
If your find our work useful, please consider citing: 


```bibtex
@misc{tesema2025denoisingwhilecompletingnetworkdwcnetrobust,
  title={Denoising-While-Completing Network (DWCNet): Robust Point Cloud Completion Under Corruption}, 
  author={Keneni W. Tesema and Lyndon Hill and Mark W. Jones and Gary K. L. Tam},
  year={2025},
  eprint={2507.16743},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.16743}, 
}

## Acknowledgement
Some codes are borrowed from PoinTr and GrNet. 





