# UltraGCN

This is our Pytorch implementation for our CIKM 2021 paper:

> Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, Xiuqiang He. UltraGCN: An Ultra Simplification of Graph Convolutional Networks for Recommendation. [Paper in arXiv](https://arxiv.org/pdf/2110.15114.pdf).



## Introduction
In this work, we propose an ultra-simplified formulation of GCN, dubbed UltraGCN. UltraGCN skips explicit message passing and directly approximate the limit of infinite message passing layers.


## Environment Requirement
The required packages are as follows:
* python: 3.7.9
* pytorch 1.4.0
* numpy: 1.19.2
* scipy: 1.1.0
* tensorboard: 2.4.0

## Reproduction
See _benchmarks_ folder to reproduce the results.
For example, we show the detail reproduce steps for the results of UltraGCN on AmazoonBooks dataset in _UltraGCN_amazonbooks_x0.md_ file.



## Results
|   Model  | AmazonBooks | AmazonBooks        |  Yelp2018 | Yelp2018        |  Gowalla  |   Gowalla      |
|:--------:|:-----------:|:-------:|:---------:|:-------:|:---------:|:-------:|
|          |  Recall@20  | nDCG@20 | Recall@20 | nDCG@20 | Recall@20 | nDCG@20 |
|   NGCF   |    0.0344   |  0.0263 |   0.0579  |  0.0477 |   0.1570  |  0.1327 |
| LightGCN |    0.0411   |  0.0315 |   0.0649  |  0.0530 |   0.1830  |  0.1554 |
| **UltraGCN** |    **0.0681**   |  **0.0556** |   **0.0683**  | **0.0561**  |   **0.1862**  |  **0.1580** |
