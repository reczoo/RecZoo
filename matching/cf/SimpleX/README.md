# SimpleX

SimpleX is a simple and strong baseline model for collaborative filtering tasks. This repo provides the official open-source implementation of our paper:

+ Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, Xiuqiang He. [SimpleX: A Simple and Strong Baseline for Collaborative Filtering](https://arxiv.org/abs/2109.12613), in CIKM 2021.

## Model Structure

SimpleX presents a simple unified CF model, which follows the commonly-used two-tower network structure to support efficient retrieval from a large item corpus. The user tower outputs a weighted combination of user profile embedding and aggregated behavior sequence embedding. The model structure is general, and with appropriate settings, it can instantiate related models such as MF, YouTubeNet, and one-hop GNN. Based on the model, we evaluate the effectiveness of cosine contrastive loss and negative sampling. 

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/reczoo/RecZoo@main/matching/cf/SimpleX/img/simplex.png" width="320" alt="SimpleX model"/>
</div>


## Environments

Our experiments were conducted in the following environment settings. For reproducibility, please follow the instructions [#32](https://github.com/reczoo/RecZoo/discussions/32) to install the dependent packages.

+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G
  ```

+ Software

  ```python
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  faiss-cpu: 1.7.0
  recbox: 0.0.4
  ```

## Configuration Guide

+ Dataset config

    ```python
    data_root: ./data/Yelp/  # data directory to save h5 data
    data_format: csv  # input data format
    train_data: ./data/Yelp/Yelp18_m1/train.csv  # training data path
    valid_data: ./data/Yelp/Yelp18_m1/test.csv  # validation data path
    test_data: ./data/Yelp/Yelp18_m1/test.csv  # test data path
    item_corpus: ./data/Yelp/Yelp18_m1/item_corpus.csv  # item corpus which maps corpus_index to item features
    min_categr_count: 1  # min count to filter category features, 
                         # e.g., features of less than 10 occurrences may be set to a default "OOV" token
    query_index: query_index  # query_index to group metrics per request/user
    corpus_index: corpus_index  # corpus_index used to map to item ids and features
    # feature_cols can be defined with the following keys:
    #     name: feature column name in csv
    #     active: True/False, whether to use the feature
    #     dtype: int/str, the input data dtype
    #     type: "index"/"categorical"/"sequence", types of features
    #     source: "user"/"item"/"context" (optional), used to group features
    #     splitter: (optional) the seperator used to split str sequence
    #     max_len: (optional) the max length to chunk or pad sequence feature
    #     padding: "pre"/"post" (optional), whether to pad before or after the original sequence
    #     embedding_callback: (optional) "layers.MaskedAveragePooling()" is used by default.
    #                         When set to "null", the sequence embedding output will not be aggregated.
    #     share_embedding: (optional) specify which features to share embedding table
    feature_cols:
        - {'name': 'query_index', 'active': True, 'dtype': int, 'type': 'index'}
        - {'name': 'corpus_index', 'active': True, 'dtype': int, 'type': 'index'}
        - {'name': 'user_id', 'active': True, 'dtype': str, 'type': 'categorical', 'source': 'user'}
        - {'name': 'user_history', 'active': True, 'dtype': str, 'type': 'sequence', 'source': 'user', 'splitter': '^',
           'max_len': 500, 'padding': 'pre', 'embedding_callback': null}
        - {'name': 'item_id', 'active': True, 'dtype': str, 'type': 'categorical', 'source': 'item', 'share_embedding': 'user_history'}
    label_col: {name: label, dtype: float}  # specify label column name and dtype
    ```

+ Model config  

    ```python
    model: SimpleX  # model class name
    dataset_id: yelp18_m1_9217a019  # dataset id to join data config
    metrics: ['Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)', 'HitRate(k=50)'] # metrics for evaluation
    optimizer: adam  # optimizer set to adam by default
    learning_rate: 1.e-4  # learning rate
    batch_size: 512 
    num_negs: 1000  # number of samples for negative sampling
    embedding_dim: 64  
    aggregator: mean  # behavior aggregator: mean/user_attention/self_attention
    gamma: 1  # combination weight g
    user_id_field: user_id  
    item_id_field: item_id  
    user_history_field: user_history  # behavior sequence
    embedding_regularizer: 1.e-8  # L2 regularization weight for embedding parameters
    net_regularizer: 0  # L2 regularization weight for network parameters
    net_dropout: 0.1  # dropout rate for network
    attention_dropout: 0  # dropout rate for attention if used
    enable_bias: False  # whether to add bias term
    similarity_score: cosine  # similarity score measure: cosine/dot
    loss: CosineContrastiveLoss  # loss used in training
    margin: 0.9  # the margin `m` threshold for CCL
    negative_weight: 150  # negative weight `w` for CCL
    sampling_num_process: 1  # number of processes for negative sampling
    fix_sampling_seeds: False  # whether to use fixed random seeds for negative sampling
    ignore_pos_items: False  # wheter to mask out positive items during negative sampling. 
                             # When set to True, the training will become more slow, but gives better results.
    epochs: 100  # the max epochs for training. Typically, training will stop by early stopping.
    shuffle: True  # whether to shuffle data samples for training
    seed: 2019  # random seed used to ensure reproducibility
    monitor: 'Recall(k=20)'  # metrics used to monitor the evaluation results for early stopping
    monitor_mode: 'max'  # `max`/`min`, indicate the higher the better or the lower the better for the monitor metric
    ```

## Results

### Results on Yelp18

|    Model    |  Recall@20 |   NDCG@20  |
|:-----------|:----------:|:----------:|
|  YouTubeNet [[RecSys'16](https://dl.acm.org/doi/10.1145/2959100.2959190)] |   0.0686   |   0.0567   |
|   ENMF [[TOIS'20](https://github.com/chenchongthu/ENMF)]       |  0.0650  |  0.0515 | 
|     NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]    |   0.0579   |   0.0477   |
|   LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)]  |   0.0649   |   0.0530   |
|  SGL-ED [[SIGIR'21](https://arxiv.org/pdf/2010.10783.pdf)]  |   0.0675  |   0.0555   |
|   UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]  |   0.0683   |   0.0561   |
| MF-CCL [[CIKM'21](https://arxiv.org/abs/2109.12613)] | 0.0698 | 0.0572 |
| SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] | **0.0701** | **0.0575** |

+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/Yelp/Yelp18_m1
    python convert_yelp18_m1.py

    # run the model
    python run_expid.py --config ./config/MF_CCL_yelp18_m1 --expid MF_CCL_yelp18_m1 --gpu 0
    python run_expid.py --config ./config/SimpleX_yelp18_m1 --expid SimpleX_yelp18_m1 --gpu 0
    ```

+ See the running logs: 
    - [results/MF_CCL_yelp18_m1_001_ab04e533.log](./results/MF_CCL_yelp18_m1_001_ab04e533.log) 
    - [results/SimpleX_yelp18_m1_034_297a4b82.log](./results/SimpleX_yelp18_m1_034_297a4b82.log)

### Results on Gowalla

|    Model   |  Recall@20 |   NDCG@20  |
|:----------|:----------:|:----------:|
| YouTubeNet [[RecSys'16](https://dl.acm.org/doi/10.1145/2959100.2959190)] |   0.1754   |   0.1473   |
|   ENMF [[TOIS'20](https://github.com/chenchongthu/ENMF)]       |  0.1523  |  0.1315 | 
|    NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]    |   0.1570   |   0.1327   |
|  LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)]  |   0.1830   |   0.1554   |
|  UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]  |   0.1862   | **0.1580** |
|   MF-CCL [[CIKM'21](https://arxiv.org/abs/2109.12613)] |   0.1837   |   0.1493   |
|   SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] | **0.1872** |   0.1557   |


+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/Gowalla/Gowalla_m1
    python convert_gowalla_m1.py

    # run the model
    python run_expid.py --config ./config/MF_CCL_gowalla_m1 --expid MF_CCL_gowalla_m1 --gpu 0
    python run_expid.py --config ./config/SimpleX_gowalla_m1 --expid SimpleX_gowalla_m1 --gpu 0
    ```

+ See the running logs: 
    - [results/MF_CCL_gowalla_m1_001_e5f1ed4e.log](./results/MF_CCL_gowalla_m1_001_e5f1ed4e.log) 
    - [results/SimpleX_gowalla_m1_013_4ecb0cbe.log](./results/SimpleX_gowalla_m1_013_4ecb0cbe.log)


### Results on Amazon-Books

|    Model   |  Recall@20 |   NDCG@20  |
|:----------|:----------:|:----------:|
| YouTubeNet [[RecSys'16](https://dl.acm.org/doi/10.1145/2959100.2959190)] |   0.0502   |   0.0388   |
|   ENMF [[TOIS'20](https://github.com/chenchongthu/ENMF)]       | 0.0359  |  0.0281  | 
|    NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]    |   0.0344   |   0.0263   |
|  LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)]  |   0.0411   |   0.0315   |
|  SGL-ED [[SIGIR'21](https://arxiv.org/pdf/2010.10783.pdf)]  |   0.0478   |   0.0379   |
|  UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)]  | **0.0681** | **0.0556** |
|   MF-CCL [[CIKM'21](https://arxiv.org/abs/2109.12613)] |   0.0559   |   0.0447   |
|   SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] |   0.0583   |   0.0468   |


+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/Amazon/AmazonBooks_m1
    python convert_amazonbooks_m1.py

    # run the model
    python run_expid.py --config ./config/MF_CCL_amazonbooks_m1 --expid MF_CCL_amazonbooks_m1 --gpu 0
    python run_expid.py --config ./config/SimpleX_amazonbooks_m1 --expid SimpleX_amazonbooks_m1 --gpu 0
    ```

+ See the running logs: 
    - [results/MF_CCL_amazonbooks_m1_001_da43988f.log](./results/MF_CCL_amazonbooks_m1_001_da43988f.log) 
    - [results/SimpleX_amazonbooks_m1_003_a30a8992.log](./results/SimpleX_amazonbooks_m1_003_a30a8992.log)



### Results on Amazon-CDs

|  Model  |  Recall@20 |   NDCG@20  |
|:-------|:----------:|:----------:|
|   NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]  |   0.1258   |   0.0792   |
|   BGCF [[KDD'20](https://dl.acm.org/doi/10.1145/3394486.3403254)]  |   0.1506   |   0.0948   |
| SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] | **0.1763** | **0.1145** |


+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/Amazon/AmazonCDs_m1
    python convert_amazoncds_m1.py

    # run the model
    python run_expid.py --config ./config/SimpleX_amazoncds_m1 --expid SimpleX_amazoncds_m1 --gpu 0
    ```

+ See the running log: [results/SimpleX_amazoncds_m1_014_c5143710.log](./results/SimpleX_amazoncds_m1_014_c5143710.log)



### Results on Amazon-Movies

|  Model  |  Recall@20 |   NDCG@20  |
|:-------|:----------:|:----------:|
|   NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]  |   0.0866   |   0.0555   |
|   BGCF [[KDD'20](https://dl.acm.org/doi/10.1145/3394486.3403254)]  |   0.1066   |   0.0693   |
| SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] | **0.1342** | **0.0926** |


+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/Amazon/AmazonMovies_m1
    python convert_amazonmovies_m1.py

    # run the model
    python run_expid.py --config ./config/SimpleX_amazonmovies_m1 --expid SimpleX_amazonmovies_m1 --gpu 0
    ```

+ See the running log: [results/SimpleX_amazonmovies_m1_009_88b07f96.log](./results/SimpleX_amazonmovies_m1_009_88b07f96.log)


### Results on Amazon-Beauty

|  Model  |  Recall@20 |   NDCG@20  |
|:-------|:----------:|:----------:|
|   NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]  |   0.1513   |   0.0917   |
|   BGCF [[KDD'20](https://dl.acm.org/doi/10.1145/3394486.3403254)]  |   0.1534   |   0.0912   |
| SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] | **0.1721** | **0.1028** |

+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/Amazon/AmazonBeauty_m1
    python convert_amazonbeauty_m1.py

    # run the model
    python run_expid.py --config ./config/SimpleX_amazonbeauty_m1 --expid SimpleX_amazonbeauty_m1 --gpu 0
    ```

+ See the running log: [results/SimpleX_amazonbeauty_m1_001_bcec104e.log](./results/SimpleX_amazonbeauty_m1_001_bcec104e.log)



### Results on Amazon-Electronics

|  Model  |    F1@20   |   NDCG@20  |
|:-------|:----------:|:----------:|
|   ENMF [[TOIS'20](https://github.com/chenchongthu/ENMF)]  |   0.0314   |   0.0823   |
|   NBPO [[SIGIR'20](https://dl.acm.org/doi/10.1145/3397271.3401155)]  |   0.0313   |   0.0810   |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)] | 0.0330 | 0.0829 |
| SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] | **0.0338** | **0.0842** |

+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/Amazon/AmazonElectronics_m1
    python convert_amazonelectronics_m1.py

    # run the model
    python run_expid.py --config ./config/SimpleX_amazonelectronics_m1 --expid SimpleX_amazonelectronics_m1 --gpu 0
    ```

+ See the running log: [results/SimpleX_amazonelectronics_m1_110_a6e18467.log](./results/SimpleX_amazonelectronics_m1_110_a6e18467.log)


### Results on CiteUlike-A

|  Model  | Precision@20 |  Recall@20 |
|:-------|:------------:|:----------:|
|   ENMF [[TOIS'20](https://github.com/chenchongthu/ENMF)]  |    0.0748    | **0.0280** |
|   NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]  |    0.0517    |   0.0193   |
|   DHCF [[KDD'20](https://dl.acm.org/doi/10.1145/3394486.3403253)]  |    0.0635    |   0.0249   |
| SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] |  **0.0754**  |   0.0269   |

+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/CiteULike/CiteUlikeA_m1
    python convert_citeulikea_m1.py

    # run the model
    python run_expid.py --config ./config/SimpleX_citeulikea_m1 --expid SimpleX_citeulikea_m1 --gpu 0
    ```

+ See the running log: [results/SimpleX_citeulikea_m1_005_fe2b7f3d.log](./results/SimpleX_citeulikea_m1_005_fe2b7f3d.log)


### Results on Movielens-1M

|   Model  |    F1@20   |   NDCG@20  |  Recall@20 |
|:--------|:----------:|:----------:|:----------:|
|   ENMF [[TOIS'20](https://github.com/chenchongthu/ENMF)]   |   0.1640   |   0.2656   |            |
|   NGCF [[SIGIR'19](https://arxiv.org/abs/1905.08108)]   |   0.1582   |   0.2511   |   0.2513   |
|   LCFN [[ICML'20](https://arxiv.org/abs/2006.15516)]   |   0.1625   |   0.2603   |            |
| LightGCN [[SIGIR'20](https://arxiv.org/abs/2002.02126)] |            |   0.2427   |   0.2576   |
| UltraGCN [[CIKM'21](https://arxiv.org/abs/2110.15114)] |    **0.2004**        |   0.2642   |   0.2787   |
|  SimpleX [[CIKM'21](https://arxiv.org/abs/2109.12613)] | 0.1658 | **0.2670** | **0.2802** |


+ Follow the steps below to reproduce the results

    ```bash
    # convert data format
    cd data/MovieLens/Movielens1M_m1
    python convert_movielens1m_m1.py

    # run the model
    python run_expid.py --config ./config/SimpleX_movielens1m_m1 --expid SimpleX_movielens1m_m1 --gpu 0
    ```

+ See the running log: [results/SimpleX_movielens1m_m1_021_6b1eda86.log](./results/SimpleX_movielens1m_m1_021_6b1eda86.log)

### Reproduce baseline results

For reproducing our baselines, please refer to the BARS benchmark at https://github.com/reczoo/BARS/tree/main/matching
