## UltraGCN_AmazonCDs_x0

A notebook to benchmark UltraGCN on AmazonCDs dataset.

Author: Kelong Mao, Renmin University

Edited by [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

    ```bash
    CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
    RAM: 128G
    ```
+ Software

    ```python
    python: 3.7.9
    pytorch 1.4.0
    numpy: 1.19.2
    scipy: 1.1.0
    tensorboard: 2.4.0
    ```

### Dataset
We follow the data split and preprocessing steps in BGCF[paper](https://dl.acm.org/doi/10.1145/3394486.3403254). We get the dataset from the author of BGCF.
### Code
Due to the conciseness of UltraGCN designs, we adopt the single file style to make the code clear and easy to be validated. All codes are in the file "main.py" with a configuration file "dataset_name_config.ini". The reproduction is very easy:

First, set your parameters in the file "dataset_name_config.ini". See "amazon_config.ini" for reference.


```bash
python main.py --config_file amazoncds_config.ini
```

### Results
Recall@20: 0.16216122741943766

NDCG@20 = 0.10432259412356267


### Logs
```bash
###################### UltraGCN ######################
1. Loading Configuration...
Computing \Omega for the item-item graph... 
i-i constraint matrix 0 ok
i-i constraint matrix 15000 ok
i-i constraint matrix 30000 ok
Computation \Omega OK!
store object in path = ./amazoncds_ii_neighbor_mat ok
store object in path = ./amazoncds_ii_constraint_mat ok
Load Configuration OK, show them below
Configuration:
{'embedding_dim': 64, 'ii_neighbor_num': 10, 'model_save_path': './ultragcn_amazoncds.pt', 'max_epoch': 2000, 'enable_tensorboard': True, 'initial_weight': 0.0001, 'dataset': 'amazoncds', 'gpu': '3', 'device': device(type='cuda', index=3), 'lr': 0.001, 'batch_size': 1024, 'early_stop_epoch': 15, 'w1': 1e-07, 'w2': 1.0, 'w3': 1e-07, 'w4': 1.0, 'negative_num': 1000, 'negative_weight': 1000.0, 'gamma': 0.0001, 'lambda': 1.0, 'sampling_sift_pos': False, 'test_batch_size': 2048, 'topk': 20, 'user_num': 43169, 'item_num': 35648}
Total training batches = 591
The time for epoch 0 is: train time = 00: 00: 50, test time = 00: 00: 09
Loss = 2001.46790, F1-score: 0.003979 	 Precision: 0.00244	 Recall: 0.01084	NDCG: 0.00676
The time for epoch 5 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 335.79581, F1-score: 0.004043 	 Precision: 0.00246	 Recall: 0.01126	NDCG: 0.00684
The time for epoch 10 is: train time = 00: 00: 49, test time = 00: 00: 09
Loss = 321.71384, F1-score: 0.003904 	 Precision: 0.00239	 Recall: 0.01059	NDCG: 0.00670
The time for epoch 15 is: train time = 00: 00: 50, test time = 00: 00: 10
Loss = 303.11905, F1-score: 0.003999 	 Precision: 0.00245	 Recall: 0.01095	NDCG: 0.00679
The time for epoch 20 is: train time = 00: 00: 38, test time = 00: 00: 10
Loss = 272.62122, F1-score: 0.006318 	 Precision: 0.00394	 Recall: 0.01599	NDCG: 0.01062
The time for epoch 25 is: train time = 00: 00: 50, test time = 00: 00: 10
Loss = 257.57416, F1-score: 0.012602 	 Precision: 0.00771	 Recall: 0.03447	NDCG: 0.02257
The time for epoch 30 is: train time = 00: 00: 49, test time = 00: 00: 09
Loss = 233.97667, F1-score: 0.020615 	 Precision: 0.01243	 Recall: 0.06028	NDCG: 0.03830
The time for epoch 35 is: train time = 00: 00: 50, test time = 00: 00: 10
Loss = 233.48837, F1-score: 0.029960 	 Precision: 0.01791	 Recall: 0.09151	NDCG: 0.05790
The time for epoch 40 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 209.47122, F1-score: 0.037753 	 Precision: 0.02244	 Recall: 0.11884	NDCG: 0.07514
The time for epoch 45 is: train time = 00: 00: 38, test time = 00: 00: 09
Loss = 210.46855, F1-score: 0.043255 	 Precision: 0.02560	 Recall: 0.13933	NDCG: 0.08878
The time for epoch 50 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 201.31866, F1-score: 0.046642 	 Precision: 0.02754	 Recall: 0.15225	NDCG: 0.09719
The time for epoch 51 is: train time = 00: 00: 50, test time = 00: 00: 09
Loss = 195.92297, F1-score: 0.047126 	 Precision: 0.02781	 Recall: 0.15438	NDCG: 0.09845
The time for epoch 52 is: train time = 00: 00: 50, test time = 00: 00: 10
Loss = 194.31705, F1-score: 0.047555 	 Precision: 0.02805	 Recall: 0.15603	NDCG: 0.09969
The time for epoch 53 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 193.71347, F1-score: 0.047933 	 Precision: 0.02826	 Recall: 0.15772	NDCG: 0.10091
The time for epoch 54 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 193.50995, F1-score: 0.048139 	 Precision: 0.02837	 Recall: 0.15871	NDCG: 0.10171
The time for epoch 55 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 191.76326, F1-score: 0.048214 	 Precision: 0.02841	 Recall: 0.15914	NDCG: 0.10211
The time for epoch 56 is: train time = 00: 00: 47, test time = 00: 00: 10
Loss = 189.12555, F1-score: 0.048467 	 Precision: 0.02856	 Recall: 0.16009	NDCG: 0.10256
The time for epoch 57 is: train time = 00: 00: 37, test time = 00: 00: 10
Loss = 189.41058, F1-score: 0.048602 	 Precision: 0.02864	 Recall: 0.16044	NDCG: 0.10289
The time for epoch 58 is: train time = 00: 00: 37, test time = 00: 00: 09
Loss = 188.31148, F1-score: 0.048544 	 Precision: 0.02859	 Recall: 0.16075	NDCG: 0.10327
The time for epoch 59 is: train time = 00: 00: 49, test time = 00: 00: 09
Loss = 188.08755, F1-score: 0.048832 	 Precision: 0.02872	 Recall: 0.16216	NDCG: 0.10432
The time for epoch 60 is: train time = 00: 00: 49, test time = 00: 00: 09
Loss = 184.66153, F1-score: 0.048707 	 Precision: 0.02868	 Recall: 0.16128	NDCG: 0.10395
The time for epoch 61 is: train time = 00: 00: 49, test time = 00: 00: 09
Loss = 180.17299, F1-score: 0.048576 	 Precision: 0.02861	 Recall: 0.16074	NDCG: 0.10363
The time for epoch 62 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 185.93759, F1-score: 0.048443 	 Precision: 0.02853	 Recall: 0.16046	NDCG: 0.10329
The time for epoch 63 is: train time = 00: 00: 49, test time = 00: 00: 09
Loss = 187.90297, F1-score: 0.048561 	 Precision: 0.02859	 Recall: 0.16097	NDCG: 0.10366
The time for epoch 64 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 184.77499, F1-score: 0.048341 	 Precision: 0.02846	 Recall: 0.16028	NDCG: 0.10336
The time for epoch 65 is: train time = 00: 00: 50, test time = 00: 00: 10
Loss = 180.52928, F1-score: 0.048418 	 Precision: 0.02851	 Recall: 0.16058	NDCG: 0.10343
The time for epoch 66 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 180.30885, F1-score: 0.048189 	 Precision: 0.02837	 Recall: 0.15977	NDCG: 0.10311
The time for epoch 67 is: train time = 00: 00: 48, test time = 00: 00: 10
Loss = 188.36577, F1-score: 0.048095 	 Precision: 0.02831	 Recall: 0.15958	NDCG: 0.10310
The time for epoch 68 is: train time = 00: 00: 38, test time = 00: 00: 09
Loss = 180.29767, F1-score: 0.048179 	 Precision: 0.02836	 Recall: 0.15995	NDCG: 0.10316
The time for epoch 69 is: train time = 00: 00: 38, test time = 00: 00: 09
Loss = 180.16411, F1-score: 0.047882 	 Precision: 0.02817	 Recall: 0.15939	NDCG: 0.10282
The time for epoch 70 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 181.36815, F1-score: 0.047805 	 Precision: 0.02814	 Recall: 0.15887	NDCG: 0.10246
The time for epoch 71 is: train time = 00: 00: 49, test time = 00: 00: 10
Loss = 182.05858, F1-score: 0.047763 	 Precision: 0.02811	 Recall: 0.15881	NDCG: 0.10257
The time for epoch 72 is: train time = 00: 00: 48, test time = 00: 00: 09
Loss = 180.16643, F1-score: 0.047710 	 Precision: 0.02808	 Recall: 0.15845	NDCG: 0.10246
The time for epoch 73 is: train time = 00: 00: 50, test time = 00: 00: 10
Loss = 179.25851, F1-score: 0.047627 	 Precision: 0.02802	 Recall: 0.15847	NDCG: 0.10241
The time for epoch 74 is: train time = 00: 00: 49, test time = 00: 00: 09
Loss = 179.26208, F1-score: 0.047365 	 Precision: 0.02786	 Recall: 0.15782	NDCG: 0.10188
##########################################
Early stop is triggered at 74 epochs.
Results:
best epoch = 59, best recall = 0.16216122741943766, best ndcg = 0.10432259412356267
The best model is saved at ./ultragcn_amazoncds.pt
Training end!
END


```
