# UltraGCN

This is Pytorch implementation for our CIKM 2021 paper:

> Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, Xiuqiang He. UltraGCN: An Ultra Simplification of Graph Convolutional Networks for Recommendation. [Paper in arXiv](https://arxiv.org/pdf/2110.15114.pdf).



## Introduction
In this work, we propose TagGNN, a heterogeneous graph neural network for more accurate item tagging under information retrieval scenario.


## Code
* main.py: All python code to reproduce UltraGCN
* dataset_name_config.ini: The configuration file which includes parameter settings for reproduction on the specific dataset.

```bash
python main.py --config_file dataset_config.ini
```
