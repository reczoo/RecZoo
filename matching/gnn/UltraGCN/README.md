# UltraGCN

UltraGCN is an ultra-simplified formulation of graph convolutional networks for collaborative filtering. This repo provides the official open-source implementation of our paper: 

+ Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, Xiuqiang He. [UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation](https://arxiv.org/pdf/2110.15114.pdf), in CIKM 2021.

## Model Overview

Graph Convolutional Networks (GCN) have been widely used for collaborative filtering. GCN models allow to capture higher-order  connections between users and items through its recursive message passing mechanism to aggregate neighborhood information. However, this message passing mechanism largely slows down the convergence of GCNs, especially when mini-batch sub-graph sampling is applied on large graphs. [LightGCN](https://arxiv.org/abs/2002.02126) reduces GCN models by removing feature transformations and nonlinear activations. In our work, UltraGCN was developed as an ultra-simplified formulation of GCNs, which skips explicit message passing and instead approximates infinite-layer graph convolutions using a constraint loss.

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/reczoo/RecZoo@main/matching/gnn/UltraGCN/img/ultragcn.png" width="500" alt="UltraGCN model"/>
</div>

## Environments

For reproducibility, please follow the instructions [#31](https://github.com/reczoo/RecZoo/discussions/31) to install the dependent packages. Otherwise, you may encounter [runtime errors](#FAQ).

## Code Structure

+ main.py: We organize all the code in a single file to make it easy to run UltraGCN. 
+ config/ultragcn_xxx.ini: Each configuration file specifies parameter settings for a target dataset.

## Results

### Results on Yelp18

| Model                                                     | Recall@20  | NDCG@20    |
|:--------------------------------------------------------- |:----------:|:----------:|
| NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]       | 0.0579     | 0.0477     |
| LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)]   | 0.0649     | 0.0530     |
| SGL-ED [[SIGIR'21](https://arxiv.org/pdf/2010.10783.pdf)] | 0.0675     | 0.0555     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]    | **0.0683** | **0.0561** |

+ Follow the script below to reproduce the results
  
  ```bash
  python main.py --config_file ./config/ultragcn_yelp18_m1.ini
  ```

+ See the running log: [results/ultragcn_yelp18_m1.log](./results/ultragcn_yelp18_m1.log) 

### Results on Gowalla

| Model                                                   | Recall@20  | NDCG@20    |
|:------------------------------------------------------- |:----------:|:----------:|
| NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]     | 0.1570     | 0.1327     |
| LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)] | 0.1830     | 0.1554     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]  | **0.1862** | **0.1580** |

+ Follow the script below to reproduce the results
  
  ```bash
  python main.py --config_file ./config/ultragcn_gowalla_m1.ini
  ```

+ See the running log: [results/ultragcn_gowalla_m1.log](./results/ultragcn_gowalla_m1.log) 

### Results on Amazon-Books

| Model                                                     | Recall@20  | NDCG@20    |
|:--------------------------------------------------------- |:----------:|:----------:|
| NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]       | 0.0344     | 0.0263     |
| LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)]   | 0.0411     | 0.0315     |
| SGL-ED [[SIGIR'21](https://arxiv.org/pdf/2010.10783.pdf)] | 0.0478     | 0.0379     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]    | **0.0681** | **0.0556** |

+ Follow the script below to reproduce the results
  
  ```bash
  python main.py --config_file ./config/ultragcn_amazonbooks_m1.ini
  ```

+ See the running log: [results/ultragcn_amazonbooks_m1.log](./results/ultragcn_amazonbooks_m1.log) 

### Results on Movielens-1M

| Model                                                   | F1@20      | NDCG@20    | Recall@20  |
|:------------------------------------------------------- |:----------:|:----------:|:----------:|
| NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]     | 0.1582     | 0.2511     | 0.2513     |
| LCFN [[ICML'20](https://arxiv.org/abs/2006.15516)]      | 0.1625     | 0.2603     |            |
| LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)] |            | 0.2427     | 0.2576     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]  | **0.2004** | **0.2642** | **0.2787** |

+ Follow the script below to reproduce the results
  
  ```bash
  # convert data format
  cd data/Movielens1M_m1
  python convert_data.py
  
  python main.py --config_file ./config/ultragcn_movielens1m_m1.ini
  ```

+ See the running log: [results/ultragcn_movielens1m_m1.log](./results/ultragcn_movielens1m_m1.log) 

### Results on Amazon-Electronics

| Model                                                             | F1@20      | NDCG@20    |
|:----------------------------------------------------------------- |:----------:|:----------:|
| ENMF [[TOIS'20](https://github.com/chenchongthu/ENMF)]            | 0.0314     | 0.0823     |
| NBPO [[SIGIR'20](https://dl.acm.org/doi/10.1145/3397271.3401155)] | 0.0313     | 0.0810     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]            | **0.0330** | **0.0829** |

+ Follow the script below to reproduce the results
  
  ```bash
  # convert data format
  cd data/AmazonElectronics_m1
  python convert_data.py

  python main.py --config_file ./config/ultragcn_amazonelectronics_m1.ini
  ```

+ See the running log: [results/ultragcn_amazonelectronics_m1.log](./results/ultragcn_amazonelectronics_m1.log) 

### Results on Amazon-CDs

| Model                                                           | Recall@20  | NDCG@20    |
|:--------------------------------------------------------------- |:----------:|:----------:|
| NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]             | 0.1258     | 0.0792     |
| BGCF [[KDD'20](https://dl.acm.org/doi/10.1145/3394486.3403254)] | 0.1506     | 0.0948     |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]          | **0.1622** | **0.1043** |

+ Follow the script below to reproduce the results
  
  ```bash
  python main.py --config_file ./config/ultragcn_amazoncds_m1.ini
  ```

+ See the running log: [results/ultragcn_amazoncds_m1.log](./results/ultragcn_amazoncds_m1.log)

### Reproduce baseline results

For reproducing our baselines, please refer to the BARS benchmark at https://github.com/reczoo/BARS/tree/main/matching

## FAQ

+ If your environment is not aligned with [ours](https://github.com/reczoo/RecZoo/discussions/31), you may encounter the following errors. We strongly recommend you to create the virtual environment following our instructions. Alternatively, you can try the fix [here](https://github.com/reczoo/RecZoo/pull/30) or [here](https://github.com/reczoo/RecZoo/issues/28#issuecomment-1704407967).

```
IndexError: tensors used as indices must be long, byte or bool tensors
```
```
RuntimeError: RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```
