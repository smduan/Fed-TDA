# Fed-TDA
The implementation of our paper "Fed-TDA: Federated Tabular Data Augmentation on Non-IID Data"

Other baseline methods: [Fedmix](https://github.com/smduan/FedMix), [Fedprox](https://github.com/smduan/FedProx), and [DP-FedAvg-GAN](https://github.com/smduan/HT-Fed-GAN/tree/main/dp-fedavg-gan).

origin data and synthsis data are saved in path `./data`

Run this repo:
1. generate synthetic data from Clinical dataset:

```
python clinical_TDA_syn.py
```

2. run eval script "clinical_eval.ipynb"
