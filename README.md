# Fed-TDA
The implementation of our paper [Fed-TDA: Federated Tabular Data Augmentation on Non-IID Data](https://arxiv.org/pdf/2211.13116.pdf)

Other baseline methods are as follow:

[1] FedMix: Approximation of Mixup under Mean Augmented Federated Learning| [paper](https://arxiv.org/pdf/2107.00233.pdf)| [code](https://github.com/smduan/FedMix)

[2]FEDERATED OPTIMIZATION IN HETEROGENEOUS NETWORKS |[paper](https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf)|[code](https://github.com/smduan/FedProx)

[3]Fed-TGAN: Federated learning framework for synthesizing tabular data|[paper](https://arxiv.org/pdf/2108.07927.pdf)|[code](https://github.com/smduan/Fed-TGAN)

[4]Generative models for effective ML on private, decentralized datasets|[paper](https://arxiv.org/pdf/1911.06679.pdf)|[code](https://github.com/smduan/HT-Fed-GAN/tree/main/dp-fedavg-gan)

# Usage Example

Run this repo:
1. generate synthetic data on Clinical dataset:

```
python clinical_TDA_syn.py
```

2. run script "clinical_eval.ipynb" to evaluate the performance of data augmentation

# Citing Fed-TDA
```
@article{duan2022fed,
  title={Fed-TDA: Federated Tabular Data Augmentation on Non-IID Data},
  author={Duan, Shaoming and Liu, Chuanyi and Han, Peiyi and He, Tianyu and Xu, Yifeng and Deng, Qiyuan},
  journal={arXiv preprint arXiv:2211.13116},
  year={2022}
}

```
