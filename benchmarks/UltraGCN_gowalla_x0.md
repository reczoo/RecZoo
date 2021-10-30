## UltraGCN_gowalla_x0

A notebook to benchmark UltraGCN on Gowalla dataset.

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
We follow the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).

### Code
Due to the conciseness of UltraGCN designs, we adopt the single file style to make the code clear and easy to be validated. All codes are in the file "main.py" with a configuration file "dataset_name_config.ini". The reproduction is very easy:

First, set your parameters in the file "dataset_name_config.ini". See "amazon_config.ini" for reference.


```bash
python main.py --config_file gowalla_config.ini
```

### Results
Recall@20 = 0.18636383812726343
NDCG@20 = 0.15809711198860704

### Logs
```bash
###################### UltraGCN ######################
1. Loading Configuration...
load path = ./gowalla_ii_constraint_mat object
load path = ./gowalla_ii_neighbor_mat object
Load Configuration OK, show them below
Configuration:
{'embedding_dim': 64, 'ii_neighbor_num': 10, 'model_save_path': './ultragcn_gowalla.pt', 'max_epoch': 2000, 'enable_tensorboard': True, 'initial_weight': 0.0001, 'dataset': 'gowalla', 'gpu': '3', 'device': device(type='cuda', index=3), 'lr': 0.0001, 'batch_size': 512, 'early_stop_epoch': 150, 'w1': 1e-06, 'w2': 1.0, 'w3': 1e-06, 'w4': 1.0, 'negative_num': 1500, 'negative_weight': 300.0, 'gamma': 0.0001, 'lambda': 0.0005, 'sampling_sift_pos': False, 'test_batch_size': 2048, 'topk': 20, 'user_num': 29858, 'item_num': 40981}
Total training batches = 1583
2021-01-27 11:07:17.271750: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
The time for epoch 0 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 1025.68616, F1-score: 0.019985 	 Precision: 0.01336	 Recall: 0.03966	NDCG: 0.03058
The time for epoch 5 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 136.82254, F1-score: 0.020896 	 Precision: 0.01393	 Recall: 0.04176	NDCG: 0.03178
The time for epoch 10 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 110.36822, F1-score: 0.020955 	 Precision: 0.01397	 Recall: 0.04188	NDCG: 0.03109
The time for epoch 15 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 96.76099, F1-score: 0.020949 	 Precision: 0.01395	 Recall: 0.04203	NDCG: 0.03112
The time for epoch 20 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 85.76865, F1-score: 0.020924 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03110
The time for epoch 25 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 81.46674, F1-score: 0.020854 	 Precision: 0.01389	 Recall: 0.04178	NDCG: 0.03104
The time for epoch 30 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 71.57571, F1-score: 0.020797 	 Precision: 0.01385	 Recall: 0.04177	NDCG: 0.03102
The time for epoch 35 is: train time = 00: 00: 52, test time = 00: 00: 07
Loss = 69.36876, F1-score: 0.020856 	 Precision: 0.01390	 Recall: 0.04178	NDCG: 0.03103
The time for epoch 40 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 67.45576, F1-score: 0.020805 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03101
The time for epoch 45 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.68381, F1-score: 0.020939 	 Precision: 0.01394	 Recall: 0.04203	NDCG: 0.03113
The time for epoch 50 is: train time = 00: 01: 00, test time = 00: 00: 08
Loss = 59.75965, F1-score: 0.020813 	 Precision: 0.01386	 Recall: 0.04181	NDCG: 0.03101
The time for epoch 51 is: train time = 00: 00: 57, test time = 00: 00: 07
Loss = 62.90189, F1-score: 0.020924 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03111
The time for epoch 52 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.29295, F1-score: 0.020924 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03109
The time for epoch 53 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.16942, F1-score: 0.020922 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03111
The time for epoch 54 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.77920, F1-score: 0.020797 	 Precision: 0.01385	 Recall: 0.04177	NDCG: 0.03099
The time for epoch 55 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.47419, F1-score: 0.020926 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03110
The time for epoch 56 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.12125, F1-score: 0.020826 	 Precision: 0.01387	 Recall: 0.04176	NDCG: 0.03101
The time for epoch 57 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.07578, F1-score: 0.020943 	 Precision: 0.01395	 Recall: 0.04203	NDCG: 0.03111
The time for epoch 58 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 59.26837, F1-score: 0.020920 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03110
The time for epoch 59 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.81866, F1-score: 0.020920 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03108
The time for epoch 60 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 55.46361, F1-score: 0.020926 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03110
The time for epoch 61 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.15963, F1-score: 0.020919 	 Precision: 0.01393	 Recall: 0.04201	NDCG: 0.03111
The time for epoch 62 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.73553, F1-score: 0.020797 	 Precision: 0.01385	 Recall: 0.04177	NDCG: 0.03099
The time for epoch 63 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.91062, F1-score: 0.020856 	 Precision: 0.01390	 Recall: 0.04178	NDCG: 0.03104
The time for epoch 64 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.97335, F1-score: 0.020939 	 Precision: 0.01394	 Recall: 0.04203	NDCG: 0.03113
The time for epoch 65 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 58.65840, F1-score: 0.020945 	 Precision: 0.01395	 Recall: 0.04203	NDCG: 0.03113
The time for epoch 66 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.64624, F1-score: 0.020797 	 Precision: 0.01385	 Recall: 0.04177	NDCG: 0.03102
The time for epoch 67 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 59.96894, F1-score: 0.020815 	 Precision: 0.01386	 Recall: 0.04181	NDCG: 0.03101
The time for epoch 68 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 57.98296, F1-score: 0.020795 	 Precision: 0.01384	 Recall: 0.04176	NDCG: 0.03101
The time for epoch 69 is: train time = 00: 00: 51, test time = 00: 00: 09
Loss = 55.48761, F1-score: 0.020807 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03103
The time for epoch 70 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.09076, F1-score: 0.020943 	 Precision: 0.01395	 Recall: 0.04203	NDCG: 0.03112
The time for epoch 71 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.26117, F1-score: 0.020852 	 Precision: 0.01389	 Recall: 0.04178	NDCG: 0.03102
The time for epoch 72 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.12236, F1-score: 0.020850 	 Precision: 0.01389	 Recall: 0.04178	NDCG: 0.03103
The time for epoch 73 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 54.82842, F1-score: 0.020949 	 Precision: 0.01395	 Recall: 0.04203	NDCG: 0.03111
The time for epoch 74 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.09880, F1-score: 0.020926 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03109
The time for epoch 75 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 53.93854, F1-score: 0.020811 	 Precision: 0.01385	 Recall: 0.04181	NDCG: 0.03101
The time for epoch 76 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.10963, F1-score: 0.020811 	 Precision: 0.01385	 Recall: 0.04181	NDCG: 0.03103
The time for epoch 77 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.38357, F1-score: 0.020809 	 Precision: 0.01385	 Recall: 0.04181	NDCG: 0.03102
The time for epoch 78 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 54.24386, F1-score: 0.020858 	 Precision: 0.01390	 Recall: 0.04178	NDCG: 0.03103
The time for epoch 79 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.31411, F1-score: 0.020933 	 Precision: 0.01394	 Recall: 0.04202	NDCG: 0.03112
The time for epoch 80 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.57463, F1-score: 0.020809 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03101
The time for epoch 81 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.69960, F1-score: 0.020926 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03114
The time for epoch 82 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.52026, F1-score: 0.020920 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03109
The time for epoch 83 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 57.85581, F1-score: 0.020939 	 Precision: 0.01394	 Recall: 0.04203	NDCG: 0.03113
The time for epoch 84 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 52.34892, F1-score: 0.020897 	 Precision: 0.01391	 Recall: 0.04199	NDCG: 0.03108
The time for epoch 85 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.85011, F1-score: 0.020850 	 Precision: 0.01389	 Recall: 0.04178	NDCG: 0.03102
The time for epoch 86 is: train time = 00: 00: 49, test time = 00: 00: 07
Loss = 57.34184, F1-score: 0.020897 	 Precision: 0.01391	 Recall: 0.04199	NDCG: 0.03107
The time for epoch 87 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.67940, F1-score: 0.020854 	 Precision: 0.01389	 Recall: 0.04178	NDCG: 0.03104
The time for epoch 88 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.24080, F1-score: 0.020762 	 Precision: 0.01382	 Recall: 0.04176	NDCG: 0.03098
The time for epoch 89 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 53.87917, F1-score: 0.020901 	 Precision: 0.01391	 Recall: 0.04199	NDCG: 0.03108
The time for epoch 90 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.04875, F1-score: 0.020922 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03111
The time for epoch 91 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.00079, F1-score: 0.020922 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03111
The time for epoch 92 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.15244, F1-score: 0.020924 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03111
The time for epoch 93 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.94194, F1-score: 0.020809 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03104
The time for epoch 94 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.21771, F1-score: 0.020809 	 Precision: 0.01385	 Recall: 0.04181	NDCG: 0.03100
The time for epoch 95 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 53.65615, F1-score: 0.020810 	 Precision: 0.01386	 Recall: 0.04170	NDCG: 0.03100
The time for epoch 96 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 52.57700, F1-score: 0.020924 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03109
The time for epoch 97 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.22878, F1-score: 0.020928 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03111
The time for epoch 98 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.48973, F1-score: 0.020808 	 Precision: 0.01386	 Recall: 0.04170	NDCG: 0.03101
The time for epoch 99 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 55.42244, F1-score: 0.020807 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03100
The time for epoch 100 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 51.98491, F1-score: 0.020805 	 Precision: 0.01385	 Recall: 0.04177	NDCG: 0.03099
The time for epoch 101 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.92204, F1-score: 0.020754 	 Precision: 0.01381	 Recall: 0.04173	NDCG: 0.03096
The time for epoch 102 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.35498, F1-score: 0.020807 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03099
The time for epoch 103 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.50722, F1-score: 0.020922 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03109
The time for epoch 104 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.79570, F1-score: 0.020775 	 Precision: 0.01383	 Recall: 0.04174	NDCG: 0.03097
The time for epoch 105 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.35472, F1-score: 0.020767 	 Precision: 0.01382	 Recall: 0.04177	NDCG: 0.03096
The time for epoch 106 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 55.57468, F1-score: 0.020811 	 Precision: 0.01385	 Recall: 0.04181	NDCG: 0.03177
The time for epoch 107 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 52.75958, F1-score: 0.020809 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03102
The time for epoch 108 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.40743, F1-score: 0.020924 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03110
The time for epoch 109 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.72612, F1-score: 0.020945 	 Precision: 0.01395	 Recall: 0.04203	NDCG: 0.03114
The time for epoch 110 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.95174, F1-score: 0.020917 	 Precision: 0.01393	 Recall: 0.04201	NDCG: 0.03112
The time for epoch 111 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.82168, F1-score: 0.020941 	 Precision: 0.01394	 Recall: 0.04205	NDCG: 0.03120
The time for epoch 112 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 55.92389, F1-score: 0.020748 	 Precision: 0.01381	 Recall: 0.04172	NDCG: 0.03094
The time for epoch 113 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 49.98186, F1-score: 0.020763 	 Precision: 0.01382	 Recall: 0.04176	NDCG: 0.03099
The time for epoch 114 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.77243, F1-score: 0.020921 	 Precision: 0.01393	 Recall: 0.04201	NDCG: 0.03185
The time for epoch 115 is: train time = 00: 00: 52, test time = 00: 00: 07
Loss = 57.76679, F1-score: 0.020901 	 Precision: 0.01391	 Recall: 0.04199	NDCG: 0.03110
The time for epoch 116 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.01099, F1-score: 0.020928 	 Precision: 0.01393	 Recall: 0.04202	NDCG: 0.03114
The time for epoch 117 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.73658, F1-score: 0.020811 	 Precision: 0.01385	 Recall: 0.04181	NDCG: 0.03286
The time for epoch 118 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 60.61808, F1-score: 0.020803 	 Precision: 0.01385	 Recall: 0.04177	NDCG: 0.03101
The time for epoch 119 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 54.42759, F1-score: 0.020858 	 Precision: 0.01390	 Recall: 0.04178	NDCG: 0.03104
The time for epoch 120 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.81786, F1-score: 0.020858 	 Precision: 0.01390	 Recall: 0.04178	NDCG: 0.03104
The time for epoch 121 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.73664, F1-score: 0.020797 	 Precision: 0.01385	 Recall: 0.04177	NDCG: 0.03103
The time for epoch 122 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.61142, F1-score: 0.020805 	 Precision: 0.01385	 Recall: 0.04180	NDCG: 0.03109
The time for epoch 123 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 55.43706, F1-score: 0.020823 	 Precision: 0.01386	 Recall: 0.04184	NDCG: 0.03114
The time for epoch 124 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.84028, F1-score: 0.020940 	 Precision: 0.01394	 Recall: 0.04202	NDCG: 0.03116
The time for epoch 125 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 55.34317, F1-score: 0.020825 	 Precision: 0.01387	 Recall: 0.04181	NDCG: 0.03107
The time for epoch 126 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.26262, F1-score: 0.020995 	 Precision: 0.01398	 Recall: 0.04210	NDCG: 0.03121
The time for epoch 127 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.61805, F1-score: 0.020909 	 Precision: 0.01392	 Recall: 0.04199	NDCG: 0.03119
The time for epoch 128 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.16037, F1-score: 0.020948 	 Precision: 0.01395	 Recall: 0.04202	NDCG: 0.03133
The time for epoch 129 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 51.91745, F1-score: 0.020882 	 Precision: 0.01391	 Recall: 0.04186	NDCG: 0.03113
The time for epoch 130 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.00398, F1-score: 0.020840 	 Precision: 0.01388	 Recall: 0.04178	NDCG: 0.03115
The time for epoch 131 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.84496, F1-score: 0.020978 	 Precision: 0.01398	 Recall: 0.04199	NDCG: 0.03164
The time for epoch 132 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 52.80840, F1-score: 0.020966 	 Precision: 0.01398	 Recall: 0.04193	NDCG: 0.03125
The time for epoch 133 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 54.06082, F1-score: 0.021027 	 Precision: 0.01402	 Recall: 0.04206	NDCG: 0.03151
The time for epoch 134 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 53.61086, F1-score: 0.020934 	 Precision: 0.01396	 Recall: 0.04186	NDCG: 0.03150
The time for epoch 135 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.46517, F1-score: 0.021052 	 Precision: 0.01405	 Recall: 0.04197	NDCG: 0.03172
The time for epoch 136 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 53.00941, F1-score: 0.021502 	 Precision: 0.01434	 Recall: 0.04295	NDCG: 0.03517
The time for epoch 137 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.50322, F1-score: 0.021248 	 Precision: 0.01420	 Recall: 0.04221	NDCG: 0.03227
The time for epoch 138 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.30191, F1-score: 0.021705 	 Precision: 0.01449	 Recall: 0.04320	NDCG: 0.03307
The time for epoch 139 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.63697, F1-score: 0.022057 	 Precision: 0.01471	 Recall: 0.04406	NDCG: 0.03415
The time for epoch 140 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 57.55938, F1-score: 0.022315 	 Precision: 0.01489	 Recall: 0.04449	NDCG: 0.03559
The time for epoch 141 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.01366, F1-score: 0.022050 	 Precision: 0.01474	 Recall: 0.04377	NDCG: 0.03458
The time for epoch 142 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 52.05024, F1-score: 0.022248 	 Precision: 0.01491	 Recall: 0.04380	NDCG: 0.03684
The time for epoch 143 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 54.22283, F1-score: 0.022475 	 Precision: 0.01509	 Recall: 0.04401	NDCG: 0.03528
The time for epoch 144 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.59992, F1-score: 0.022701 	 Precision: 0.01527	 Recall: 0.04426	NDCG: 0.03663
The time for epoch 145 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.43021, F1-score: 0.023144 	 Precision: 0.01552	 Recall: 0.04546	NDCG: 0.03674
The time for epoch 146 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 51.75937, F1-score: 0.023349 	 Precision: 0.01570	 Recall: 0.04551	NDCG: 0.03856
The time for epoch 147 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 53.17975, F1-score: 0.023721 	 Precision: 0.01598	 Recall: 0.04603	NDCG: 0.03976
The time for epoch 148 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 54.11795, F1-score: 0.024037 	 Precision: 0.01623	 Recall: 0.04633	NDCG: 0.04000
The time for epoch 149 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.33044, F1-score: 0.024511 	 Precision: 0.01657	 Recall: 0.04705	NDCG: 0.04017
The time for epoch 150 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.93228, F1-score: 0.024979 	 Precision: 0.01693	 Recall: 0.04764	NDCG: 0.04167
The time for epoch 151 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 52.28700, F1-score: 0.025480 	 Precision: 0.01726	 Recall: 0.04866	NDCG: 0.04247
The time for epoch 152 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.96621, F1-score: 0.026021 	 Precision: 0.01765	 Recall: 0.04953	NDCG: 0.04353
The time for epoch 153 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 53.57205, F1-score: 0.026542 	 Precision: 0.01802	 Recall: 0.05033	NDCG: 0.04452
The time for epoch 154 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.34880, F1-score: 0.026995 	 Precision: 0.01836	 Recall: 0.05100	NDCG: 0.04564
The time for epoch 155 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.23626, F1-score: 0.027579 	 Precision: 0.01877	 Recall: 0.05196	NDCG: 0.04666
The time for epoch 156 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.63549, F1-score: 0.028117 	 Precision: 0.01915	 Recall: 0.05285	NDCG: 0.04806
The time for epoch 157 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 55.55978, F1-score: 0.028562 	 Precision: 0.01949	 Recall: 0.05347	NDCG: 0.04860
The time for epoch 158 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 51.98218, F1-score: 0.029210 	 Precision: 0.01995	 Recall: 0.05454	NDCG: 0.05042
The time for epoch 159 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 53.02409, F1-score: 0.029763 	 Precision: 0.02034	 Recall: 0.05542	NDCG: 0.05167
The time for epoch 160 is: train time = 00: 00: 58, test time = 00: 00: 09
Loss = 52.80465, F1-score: 0.030420 	 Precision: 0.02080	 Recall: 0.05657	NDCG: 0.05255
The time for epoch 161 is: train time = 00: 00: 58, test time = 00: 00: 08
Loss = 53.61193, F1-score: 0.031152 	 Precision: 0.02132	 Recall: 0.05778	NDCG: 0.05367
The time for epoch 162 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 54.72514, F1-score: 0.031829 	 Precision: 0.02178	 Recall: 0.05907	NDCG: 0.05520
The time for epoch 163 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 54.57094, F1-score: 0.032463 	 Precision: 0.02223	 Recall: 0.06019	NDCG: 0.05673
The time for epoch 164 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 51.52998, F1-score: 0.033158 	 Precision: 0.02269	 Recall: 0.06158	NDCG: 0.05777
The time for epoch 165 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.32373, F1-score: 0.033844 	 Precision: 0.02314	 Recall: 0.06296	NDCG: 0.05913
The time for epoch 166 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 55.41048, F1-score: 0.034553 	 Precision: 0.02361	 Recall: 0.06442	NDCG: 0.06021
The time for epoch 167 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 55.80075, F1-score: 0.035364 	 Precision: 0.02415	 Recall: 0.06601	NDCG: 0.06151
The time for epoch 168 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 55.48511, F1-score: 0.036022 	 Precision: 0.02456	 Recall: 0.06751	NDCG: 0.06286
The time for epoch 169 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.58640, F1-score: 0.036791 	 Precision: 0.02506	 Recall: 0.06915	NDCG: 0.06410
The time for epoch 170 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 54.96464, F1-score: 0.037470 	 Precision: 0.02551	 Recall: 0.07057	NDCG: 0.06552
The time for epoch 171 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.13734, F1-score: 0.038219 	 Precision: 0.02600	 Recall: 0.07208	NDCG: 0.06690
The time for epoch 172 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.45776, F1-score: 0.038891 	 Precision: 0.02645	 Recall: 0.07344	NDCG: 0.06817
The time for epoch 173 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 55.90870, F1-score: 0.039587 	 Precision: 0.02690	 Recall: 0.07493	NDCG: 0.06934
The time for epoch 174 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 52.97748, F1-score: 0.040401 	 Precision: 0.02742	 Recall: 0.07674	NDCG: 0.07087
The time for epoch 175 is: train time = 00: 00: 51, test time = 00: 00: 07
Loss = 53.53628, F1-score: 0.041149 	 Precision: 0.02790	 Recall: 0.07834	NDCG: 0.07213
The time for epoch 176 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.23479, F1-score: 0.041963 	 Precision: 0.02843	 Recall: 0.08010	NDCG: 0.07357
The time for epoch 177 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.91676, F1-score: 0.042555 	 Precision: 0.02878	 Recall: 0.08159	NDCG: 0.07488
The time for epoch 178 is: train time = 00: 00: 50, test time = 00: 00: 07
Loss = 52.47895, F1-score: 0.043391 	 Precision: 0.02931	 Recall: 0.08349	NDCG: 0.07638
The time for epoch 179 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.25061, F1-score: 0.044147 	 Precision: 0.02978	 Recall: 0.08527	NDCG: 0.07787
The time for epoch 180 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.26316, F1-score: 0.044962 	 Precision: 0.03029	 Recall: 0.08721	NDCG: 0.07955
The time for epoch 181 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.07644, F1-score: 0.045557 	 Precision: 0.03065	 Recall: 0.08870	NDCG: 0.08093
The time for epoch 182 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 51.98526, F1-score: 0.046286 	 Precision: 0.03110	 Recall: 0.09048	NDCG: 0.08240
The time for epoch 183 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.54889, F1-score: 0.047037 	 Precision: 0.03158	 Recall: 0.09215	NDCG: 0.08375
The time for epoch 184 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.77032, F1-score: 0.047563 	 Precision: 0.03190	 Recall: 0.09344	NDCG: 0.08502
The time for epoch 185 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.51465, F1-score: 0.048224 	 Precision: 0.03230	 Recall: 0.09512	NDCG: 0.08621
The time for epoch 186 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 53.52616, F1-score: 0.048948 	 Precision: 0.03276	 Recall: 0.09680	NDCG: 0.08786
The time for epoch 187 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.80067, F1-score: 0.049488 	 Precision: 0.03308	 Recall: 0.09822	NDCG: 0.08864
The time for epoch 188 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.83319, F1-score: 0.049994 	 Precision: 0.03337	 Recall: 0.09958	NDCG: 0.09001
The time for epoch 189 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.45552, F1-score: 0.050524 	 Precision: 0.03371	 Recall: 0.10082	NDCG: 0.09107
The time for epoch 190 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 54.15576, F1-score: 0.051104 	 Precision: 0.03408	 Recall: 0.10215	NDCG: 0.09234
The time for epoch 191 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.49636, F1-score: 0.051618 	 Precision: 0.03439	 Recall: 0.10339	NDCG: 0.09334
The time for epoch 192 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 60.06546, F1-score: 0.052132 	 Precision: 0.03471	 Recall: 0.10464	NDCG: 0.09430
The time for epoch 193 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.36007, F1-score: 0.052528 	 Precision: 0.03497	 Recall: 0.10554	NDCG: 0.09524
The time for epoch 194 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.73967, F1-score: 0.053030 	 Precision: 0.03528	 Recall: 0.10677	NDCG: 0.09612
The time for epoch 195 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.03097, F1-score: 0.053478 	 Precision: 0.03557	 Recall: 0.10772	NDCG: 0.09696
The time for epoch 196 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 57.38764, F1-score: 0.053928 	 Precision: 0.03585	 Recall: 0.10879	NDCG: 0.09793
The time for epoch 197 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 53.61590, F1-score: 0.054446 	 Precision: 0.03617	 Recall: 0.11008	NDCG: 0.09884
The time for epoch 198 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.62479, F1-score: 0.054747 	 Precision: 0.03636	 Recall: 0.11074	NDCG: 0.09941
The time for epoch 199 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 53.92485, F1-score: 0.055273 	 Precision: 0.03670	 Recall: 0.11194	NDCG: 0.10032
The time for epoch 200 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 55.29043, F1-score: 0.055719 	 Precision: 0.03699	 Recall: 0.11291	NDCG: 0.10115
The time for epoch 201 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.03210, F1-score: 0.056250 	 Precision: 0.03732	 Recall: 0.11415	NDCG: 0.10215
The time for epoch 202 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.07872, F1-score: 0.056689 	 Precision: 0.03760	 Recall: 0.11516	NDCG: 0.10302
The time for epoch 203 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 53.15860, F1-score: 0.057097 	 Precision: 0.03785	 Recall: 0.11618	NDCG: 0.10381
The time for epoch 204 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 55.19036, F1-score: 0.057602 	 Precision: 0.03817	 Recall: 0.11736	NDCG: 0.10476
The time for epoch 205 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.41660, F1-score: 0.058052 	 Precision: 0.03846	 Recall: 0.11836	NDCG: 0.10549
The time for epoch 206 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 54.55211, F1-score: 0.058460 	 Precision: 0.03872	 Recall: 0.11921	NDCG: 0.10608
The time for epoch 207 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 57.07451, F1-score: 0.058933 	 Precision: 0.03903	 Recall: 0.12024	NDCG: 0.10697
The time for epoch 208 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 55.25113, F1-score: 0.059302 	 Precision: 0.03926	 Recall: 0.12117	NDCG: 0.10770
The time for epoch 209 is: train time = 00: 00: 52, test time = 00: 00: 07
Loss = 54.96304, F1-score: 0.059594 	 Precision: 0.03944	 Recall: 0.12190	NDCG: 0.10831
The time for epoch 210 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 56.22311, F1-score: 0.060041 	 Precision: 0.03972	 Recall: 0.12295	NDCG: 0.10913
The time for epoch 211 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 53.32863, F1-score: 0.060389 	 Precision: 0.03994	 Recall: 0.12373	NDCG: 0.10982
The time for epoch 212 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.72509, F1-score: 0.060746 	 Precision: 0.04017	 Recall: 0.12455	NDCG: 0.11042
The time for epoch 213 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.62165, F1-score: 0.061074 	 Precision: 0.04039	 Recall: 0.12515	NDCG: 0.11111
The time for epoch 214 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.80342, F1-score: 0.061324 	 Precision: 0.04055	 Recall: 0.12579	NDCG: 0.11169
The time for epoch 215 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.02433, F1-score: 0.061741 	 Precision: 0.04081	 Recall: 0.12677	NDCG: 0.11233
The time for epoch 216 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.96589, F1-score: 0.062068 	 Precision: 0.04102	 Recall: 0.12747	NDCG: 0.11297
The time for epoch 217 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.83867, F1-score: 0.062428 	 Precision: 0.04125	 Recall: 0.12829	NDCG: 0.11368
The time for epoch 218 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.17285, F1-score: 0.062852 	 Precision: 0.04151	 Recall: 0.12939	NDCG: 0.11440
The time for epoch 219 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.92984, F1-score: 0.063086 	 Precision: 0.04165	 Recall: 0.12996	NDCG: 0.11490
The time for epoch 220 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 58.47380, F1-score: 0.063249 	 Precision: 0.04175	 Recall: 0.13036	NDCG: 0.11538
The time for epoch 221 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.34534, F1-score: 0.063711 	 Precision: 0.04205	 Recall: 0.13136	NDCG: 0.11605
The time for epoch 222 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.40838, F1-score: 0.063950 	 Precision: 0.04220	 Recall: 0.13193	NDCG: 0.11646
The time for epoch 223 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 59.17608, F1-score: 0.064177 	 Precision: 0.04235	 Recall: 0.13248	NDCG: 0.11712
The time for epoch 224 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.72831, F1-score: 0.064536 	 Precision: 0.04257	 Recall: 0.13329	NDCG: 0.11779
The time for epoch 225 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.78352, F1-score: 0.064717 	 Precision: 0.04270	 Recall: 0.13365	NDCG: 0.11810
The time for epoch 226 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.16658, F1-score: 0.065023 	 Precision: 0.04289	 Recall: 0.13434	NDCG: 0.11869
The time for epoch 227 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.18676, F1-score: 0.065303 	 Precision: 0.04307	 Recall: 0.13502	NDCG: 0.11911
The time for epoch 228 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.58680, F1-score: 0.065483 	 Precision: 0.04318	 Recall: 0.13541	NDCG: 0.11953
The time for epoch 229 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.87620, F1-score: 0.065645 	 Precision: 0.04328	 Recall: 0.13585	NDCG: 0.11989
The time for epoch 230 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 54.90253, F1-score: 0.066010 	 Precision: 0.04352	 Recall: 0.13656	NDCG: 0.12049
The time for epoch 231 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.00600, F1-score: 0.066191 	 Precision: 0.04364	 Recall: 0.13697	NDCG: 0.12094
The time for epoch 232 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.65088, F1-score: 0.066378 	 Precision: 0.04375	 Recall: 0.13747	NDCG: 0.12130
The time for epoch 233 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 55.92548, F1-score: 0.066676 	 Precision: 0.04394	 Recall: 0.13817	NDCG: 0.12180
The time for epoch 234 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.69520, F1-score: 0.066843 	 Precision: 0.04404	 Recall: 0.13858	NDCG: 0.12216
The time for epoch 235 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.21214, F1-score: 0.067060 	 Precision: 0.04418	 Recall: 0.13905	NDCG: 0.12246
The time for epoch 236 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.40886, F1-score: 0.067398 	 Precision: 0.04440	 Recall: 0.13982	NDCG: 0.12301
The time for epoch 237 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.39717, F1-score: 0.067554 	 Precision: 0.04450	 Recall: 0.14014	NDCG: 0.12332
The time for epoch 238 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 57.87500, F1-score: 0.067743 	 Precision: 0.04463	 Recall: 0.14055	NDCG: 0.12382
The time for epoch 239 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.71489, F1-score: 0.067928 	 Precision: 0.04475	 Recall: 0.14090	NDCG: 0.12406
The time for epoch 240 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 57.36447, F1-score: 0.068136 	 Precision: 0.04487	 Recall: 0.14150	NDCG: 0.12443
The time for epoch 241 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 58.38561, F1-score: 0.068309 	 Precision: 0.04498	 Recall: 0.14194	NDCG: 0.12483
The time for epoch 242 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.42970, F1-score: 0.068524 	 Precision: 0.04512	 Recall: 0.14238	NDCG: 0.12526
The time for epoch 243 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.10078, F1-score: 0.068753 	 Precision: 0.04528	 Recall: 0.14278	NDCG: 0.12557
The time for epoch 244 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.66301, F1-score: 0.068829 	 Precision: 0.04533	 Recall: 0.14294	NDCG: 0.12589
The time for epoch 245 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.77286, F1-score: 0.069072 	 Precision: 0.04549	 Recall: 0.14345	NDCG: 0.12633
The time for epoch 246 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 58.80780, F1-score: 0.069281 	 Precision: 0.04562	 Recall: 0.14393	NDCG: 0.12660
The time for epoch 247 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.03579, F1-score: 0.069449 	 Precision: 0.04572	 Recall: 0.14434	NDCG: 0.12688
The time for epoch 248 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.53513, F1-score: 0.069656 	 Precision: 0.04586	 Recall: 0.14476	NDCG: 0.12730
The time for epoch 249 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.70747, F1-score: 0.069882 	 Precision: 0.04600	 Recall: 0.14533	NDCG: 0.12760
The time for epoch 250 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.41119, F1-score: 0.069962 	 Precision: 0.04605	 Recall: 0.14552	NDCG: 0.12785
The time for epoch 251 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.41110, F1-score: 0.070131 	 Precision: 0.04617	 Recall: 0.14583	NDCG: 0.12808
The time for epoch 252 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 60.49470, F1-score: 0.070324 	 Precision: 0.04628	 Recall: 0.14636	NDCG: 0.12860
The time for epoch 253 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.36017, F1-score: 0.070437 	 Precision: 0.04636	 Recall: 0.14657	NDCG: 0.12878
The time for epoch 254 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.08304, F1-score: 0.070626 	 Precision: 0.04648	 Recall: 0.14695	NDCG: 0.12905
The time for epoch 255 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 55.47689, F1-score: 0.070744 	 Precision: 0.04656	 Recall: 0.14725	NDCG: 0.12931
The time for epoch 256 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.02436, F1-score: 0.070924 	 Precision: 0.04667	 Recall: 0.14762	NDCG: 0.12971
The time for epoch 257 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.11917, F1-score: 0.071189 	 Precision: 0.04684	 Recall: 0.14828	NDCG: 0.13015
The time for epoch 258 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.22494, F1-score: 0.071226 	 Precision: 0.04685	 Recall: 0.14846	NDCG: 0.13031
The time for epoch 259 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.65366, F1-score: 0.071404 	 Precision: 0.04698	 Recall: 0.14876	NDCG: 0.13053
The time for epoch 260 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.53211, F1-score: 0.071593 	 Precision: 0.04710	 Recall: 0.14915	NDCG: 0.13091
The time for epoch 261 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 59.80749, F1-score: 0.071671 	 Precision: 0.04714	 Recall: 0.14939	NDCG: 0.13112
The time for epoch 262 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.98755, F1-score: 0.071919 	 Precision: 0.04731	 Recall: 0.14993	NDCG: 0.13159
The time for epoch 263 is: train time = 00: 00: 52, test time = 00: 00: 07
Loss = 58.00913, F1-score: 0.072002 	 Precision: 0.04736	 Recall: 0.15012	NDCG: 0.13177
The time for epoch 264 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.06086, F1-score: 0.072077 	 Precision: 0.04740	 Recall: 0.15032	NDCG: 0.13198
The time for epoch 265 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 59.09291, F1-score: 0.072401 	 Precision: 0.04761	 Recall: 0.15111	NDCG: 0.13245
The time for epoch 266 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 56.42702, F1-score: 0.072454 	 Precision: 0.04764	 Recall: 0.15118	NDCG: 0.13253
The time for epoch 267 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.47673, F1-score: 0.072605 	 Precision: 0.04773	 Recall: 0.15165	NDCG: 0.13279
The time for epoch 268 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.70689, F1-score: 0.072846 	 Precision: 0.04788	 Recall: 0.15223	NDCG: 0.13315
The time for epoch 269 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 59.41909, F1-score: 0.072952 	 Precision: 0.04796	 Recall: 0.15238	NDCG: 0.13327
The time for epoch 270 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.38613, F1-score: 0.073141 	 Precision: 0.04807	 Recall: 0.15287	NDCG: 0.13362
The time for epoch 271 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.75388, F1-score: 0.073241 	 Precision: 0.04814	 Recall: 0.15301	NDCG: 0.13387
The time for epoch 272 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.23721, F1-score: 0.073392 	 Precision: 0.04823	 Recall: 0.15342	NDCG: 0.13408
The time for epoch 273 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.03309, F1-score: 0.073570 	 Precision: 0.04835	 Recall: 0.15380	NDCG: 0.13436
The time for epoch 274 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.39499, F1-score: 0.073642 	 Precision: 0.04839	 Recall: 0.15406	NDCG: 0.13460
The time for epoch 275 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 58.96750, F1-score: 0.073800 	 Precision: 0.04849	 Recall: 0.15435	NDCG: 0.13497
The time for epoch 276 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.06034, F1-score: 0.073990 	 Precision: 0.04861	 Recall: 0.15480	NDCG: 0.13517
The time for epoch 277 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 57.17816, F1-score: 0.074114 	 Precision: 0.04869	 Recall: 0.15513	NDCG: 0.13542
The time for epoch 278 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.50318, F1-score: 0.074167 	 Precision: 0.04872	 Recall: 0.15526	NDCG: 0.13567
The time for epoch 279 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.72779, F1-score: 0.074387 	 Precision: 0.04887	 Recall: 0.15571	NDCG: 0.13598
The time for epoch 280 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.77744, F1-score: 0.074459 	 Precision: 0.04891	 Recall: 0.15589	NDCG: 0.13611
The time for epoch 281 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 57.24139, F1-score: 0.074636 	 Precision: 0.04903	 Recall: 0.15627	NDCG: 0.13635
The time for epoch 282 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.84385, F1-score: 0.074834 	 Precision: 0.04915	 Recall: 0.15678	NDCG: 0.13666
The time for epoch 283 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.53471, F1-score: 0.074901 	 Precision: 0.04919	 Recall: 0.15687	NDCG: 0.13678
The time for epoch 284 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.73192, F1-score: 0.075023 	 Precision: 0.04927	 Recall: 0.15718	NDCG: 0.13710
The time for epoch 285 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.01517, F1-score: 0.075135 	 Precision: 0.04933	 Recall: 0.15752	NDCG: 0.13720
The time for epoch 286 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 57.55969, F1-score: 0.075214 	 Precision: 0.04939	 Recall: 0.15760	NDCG: 0.13737
The time for epoch 287 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.28853, F1-score: 0.075270 	 Precision: 0.04944	 Recall: 0.15760	NDCG: 0.13759
The time for epoch 288 is: train time = 00: 00: 51, test time = 00: 00: 09
Loss = 56.89935, F1-score: 0.075474 	 Precision: 0.04956	 Recall: 0.15815	NDCG: 0.13787
The time for epoch 289 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.73186, F1-score: 0.075669 	 Precision: 0.04968	 Recall: 0.15868	NDCG: 0.13820
The time for epoch 290 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.47788, F1-score: 0.075852 	 Precision: 0.04980	 Recall: 0.15910	NDCG: 0.13858
The time for epoch 291 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 57.41328, F1-score: 0.075959 	 Precision: 0.04987	 Recall: 0.15924	NDCG: 0.13873
The time for epoch 292 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.54629, F1-score: 0.075983 	 Precision: 0.04989	 Recall: 0.15927	NDCG: 0.13889
The time for epoch 293 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 56.68647, F1-score: 0.076159 	 Precision: 0.05000	 Recall: 0.15969	NDCG: 0.13921
The time for epoch 294 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.94862, F1-score: 0.076326 	 Precision: 0.05011	 Recall: 0.16008	NDCG: 0.13936
The time for epoch 295 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.63228, F1-score: 0.076371 	 Precision: 0.05012	 Recall: 0.16032	NDCG: 0.13949
The time for epoch 296 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.76403, F1-score: 0.076363 	 Precision: 0.05013	 Recall: 0.16015	NDCG: 0.13960
The time for epoch 297 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.46702, F1-score: 0.076456 	 Precision: 0.05019	 Recall: 0.16044	NDCG: 0.13972
The time for epoch 298 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.97630, F1-score: 0.076581 	 Precision: 0.05027	 Recall: 0.16072	NDCG: 0.13995
The time for epoch 299 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.83355, F1-score: 0.076703 	 Precision: 0.05033	 Recall: 0.16115	NDCG: 0.14023
The time for epoch 300 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 60.01015, F1-score: 0.076842 	 Precision: 0.05042	 Recall: 0.16144	NDCG: 0.14060
The time for epoch 301 is: train time = 00: 00: 52, test time = 00: 00: 07
Loss = 60.28590, F1-score: 0.076927 	 Precision: 0.05048	 Recall: 0.16161	NDCG: 0.14055
The time for epoch 302 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.55579, F1-score: 0.077101 	 Precision: 0.05059	 Recall: 0.16196	NDCG: 0.14089
The time for epoch 303 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.62423, F1-score: 0.077197 	 Precision: 0.05066	 Recall: 0.16210	NDCG: 0.14103
The time for epoch 304 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.42140, F1-score: 0.077190 	 Precision: 0.05066	 Recall: 0.16206	NDCG: 0.14116
The time for epoch 305 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 57.99228, F1-score: 0.077347 	 Precision: 0.05075	 Recall: 0.16253	NDCG: 0.14134
The time for epoch 306 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.46031, F1-score: 0.077396 	 Precision: 0.05079	 Recall: 0.16254	NDCG: 0.14143
The time for epoch 307 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 59.37470, F1-score: 0.077621 	 Precision: 0.05093	 Recall: 0.16312	NDCG: 0.14179
The time for epoch 308 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 60.35776, F1-score: 0.077631 	 Precision: 0.05093	 Recall: 0.16319	NDCG: 0.14190
The time for epoch 309 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.08921, F1-score: 0.077757 	 Precision: 0.05100	 Recall: 0.16354	NDCG: 0.14212
The time for epoch 310 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.23743, F1-score: 0.078026 	 Precision: 0.05118	 Recall: 0.16407	NDCG: 0.14243
The time for epoch 311 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 59.04555, F1-score: 0.078083 	 Precision: 0.05123	 Recall: 0.16414	NDCG: 0.14251
The time for epoch 312 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 56.18437, F1-score: 0.078200 	 Precision: 0.05130	 Recall: 0.16440	NDCG: 0.14274
The time for epoch 313 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.76631, F1-score: 0.078288 	 Precision: 0.05136	 Recall: 0.16456	NDCG: 0.14297
The time for epoch 314 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.91194, F1-score: 0.078258 	 Precision: 0.05134	 Recall: 0.16455	NDCG: 0.14287
The time for epoch 315 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.55993, F1-score: 0.078402 	 Precision: 0.05142	 Recall: 0.16498	NDCG: 0.14312
The time for epoch 316 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.10371, F1-score: 0.078516 	 Precision: 0.05148	 Recall: 0.16532	NDCG: 0.14330
The time for epoch 317 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.50376, F1-score: 0.078603 	 Precision: 0.05153	 Recall: 0.16557	NDCG: 0.14351
The time for epoch 318 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.12263, F1-score: 0.078713 	 Precision: 0.05161	 Recall: 0.16572	NDCG: 0.14372
The time for epoch 319 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.57471, F1-score: 0.078832 	 Precision: 0.05168	 Recall: 0.16609	NDCG: 0.14406
The time for epoch 320 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.50628, F1-score: 0.078927 	 Precision: 0.05175	 Recall: 0.16624	NDCG: 0.14409
The time for epoch 321 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.30857, F1-score: 0.079115 	 Precision: 0.05186	 Recall: 0.16673	NDCG: 0.14431
The time for epoch 322 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 58.08152, F1-score: 0.079203 	 Precision: 0.05192	 Recall: 0.16691	NDCG: 0.14448
The time for epoch 323 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.66890, F1-score: 0.079325 	 Precision: 0.05200	 Recall: 0.16719	NDCG: 0.14469
The time for epoch 324 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 57.10894, F1-score: 0.079389 	 Precision: 0.05204	 Recall: 0.16733	NDCG: 0.14485
The time for epoch 325 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 57.69676, F1-score: 0.079424 	 Precision: 0.05206	 Recall: 0.16747	NDCG: 0.14486
The time for epoch 326 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 58.90248, F1-score: 0.079388 	 Precision: 0.05205	 Recall: 0.16725	NDCG: 0.14484
The time for epoch 327 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 57.84150, F1-score: 0.079460 	 Precision: 0.05209	 Recall: 0.16747	NDCG: 0.14498
The time for epoch 328 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 59.03245, F1-score: 0.079589 	 Precision: 0.05217	 Recall: 0.16772	NDCG: 0.14509
The time for epoch 329 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.97022, F1-score: 0.079761 	 Precision: 0.05227	 Recall: 0.16822	NDCG: 0.14545
The time for epoch 330 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 56.99874, F1-score: 0.079965 	 Precision: 0.05241	 Recall: 0.16856	NDCG: 0.14568
The time for epoch 331 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.96067, F1-score: 0.079971 	 Precision: 0.05241	 Recall: 0.16862	NDCG: 0.14576
The time for epoch 332 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.12484, F1-score: 0.079990 	 Precision: 0.05243	 Recall: 0.16866	NDCG: 0.14587
The time for epoch 333 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 58.41766, F1-score: 0.080193 	 Precision: 0.05255	 Recall: 0.16919	NDCG: 0.14610
The time for epoch 334 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.40840, F1-score: 0.080268 	 Precision: 0.05261	 Recall: 0.16924	NDCG: 0.14608
The time for epoch 335 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.90381, F1-score: 0.080313 	 Precision: 0.05264	 Recall: 0.16934	NDCG: 0.14621
The time for epoch 336 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.92782, F1-score: 0.080472 	 Precision: 0.05274	 Recall: 0.16975	NDCG: 0.14655
The time for epoch 337 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.02298, F1-score: 0.080465 	 Precision: 0.05273	 Recall: 0.16974	NDCG: 0.14654
The time for epoch 338 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.87858, F1-score: 0.080596 	 Precision: 0.05281	 Recall: 0.17004	NDCG: 0.14677
The time for epoch 339 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.40995, F1-score: 0.080674 	 Precision: 0.05286	 Recall: 0.17022	NDCG: 0.14692
The time for epoch 340 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.88034, F1-score: 0.080738 	 Precision: 0.05291	 Recall: 0.17031	NDCG: 0.14715
The time for epoch 341 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.53043, F1-score: 0.080722 	 Precision: 0.05290	 Recall: 0.17032	NDCG: 0.14710
The time for epoch 342 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 59.62378, F1-score: 0.080780 	 Precision: 0.05293	 Recall: 0.17050	NDCG: 0.14721
The time for epoch 343 is: train time = 00: 00: 51, test time = 00: 00: 09
Loss = 58.89078, F1-score: 0.080887 	 Precision: 0.05299	 Recall: 0.17084	NDCG: 0.14734
The time for epoch 344 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 57.67186, F1-score: 0.081012 	 Precision: 0.05308	 Recall: 0.17103	NDCG: 0.14751
The time for epoch 345 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 60.68396, F1-score: 0.081104 	 Precision: 0.05314	 Recall: 0.17119	NDCG: 0.14758
The time for epoch 346 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.00201, F1-score: 0.081201 	 Precision: 0.05320	 Recall: 0.17146	NDCG: 0.14773
The time for epoch 347 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 59.59856, F1-score: 0.081373 	 Precision: 0.05330	 Recall: 0.17192	NDCG: 0.14814
The time for epoch 348 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 59.96198, F1-score: 0.081373 	 Precision: 0.05331	 Recall: 0.17185	NDCG: 0.14817
The time for epoch 349 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.13789, F1-score: 0.081390 	 Precision: 0.05331	 Recall: 0.17198	NDCG: 0.14827
The time for epoch 350 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.76386, F1-score: 0.081529 	 Precision: 0.05339	 Recall: 0.17235	NDCG: 0.14835
The time for epoch 351 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 58.56974, F1-score: 0.081620 	 Precision: 0.05346	 Recall: 0.17249	NDCG: 0.14848
The time for epoch 352 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.48045, F1-score: 0.081753 	 Precision: 0.05354	 Recall: 0.17286	NDCG: 0.14880
The time for epoch 353 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.04353, F1-score: 0.081818 	 Precision: 0.05359	 Recall: 0.17291	NDCG: 0.14879
The time for epoch 354 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 59.20061, F1-score: 0.081801 	 Precision: 0.05357	 Recall: 0.17294	NDCG: 0.14894
The time for epoch 355 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.82363, F1-score: 0.081861 	 Precision: 0.05361	 Recall: 0.17307	NDCG: 0.14888
The time for epoch 356 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 59.84375, F1-score: 0.081994 	 Precision: 0.05369	 Recall: 0.17344	NDCG: 0.14910
The time for epoch 357 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.80208, F1-score: 0.082022 	 Precision: 0.05371	 Recall: 0.17341	NDCG: 0.14917
The time for epoch 358 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.64067, F1-score: 0.082104 	 Precision: 0.05376	 Recall: 0.17365	NDCG: 0.14929
The time for epoch 359 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 59.24289, F1-score: 0.082175 	 Precision: 0.05381	 Recall: 0.17379	NDCG: 0.14933
The time for epoch 360 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.44212, F1-score: 0.082270 	 Precision: 0.05387	 Recall: 0.17399	NDCG: 0.14951
The time for epoch 361 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.05361, F1-score: 0.082277 	 Precision: 0.05388	 Recall: 0.17399	NDCG: 0.14961
The time for epoch 362 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.92252, F1-score: 0.082420 	 Precision: 0.05396	 Recall: 0.17445	NDCG: 0.14980
The time for epoch 363 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 61.17850, F1-score: 0.082418 	 Precision: 0.05396	 Recall: 0.17437	NDCG: 0.14991
The time for epoch 364 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.05198, F1-score: 0.082401 	 Precision: 0.05395	 Recall: 0.17437	NDCG: 0.14985
The time for epoch 365 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.49596, F1-score: 0.082482 	 Precision: 0.05401	 Recall: 0.17444	NDCG: 0.14995
The time for epoch 366 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.21214, F1-score: 0.082558 	 Precision: 0.05405	 Recall: 0.17475	NDCG: 0.15004
The time for epoch 367 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.16603, F1-score: 0.082620 	 Precision: 0.05408	 Recall: 0.17490	NDCG: 0.15020
The time for epoch 368 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.76955, F1-score: 0.082667 	 Precision: 0.05410	 Recall: 0.17511	NDCG: 0.15018
The time for epoch 369 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.95641, F1-score: 0.082781 	 Precision: 0.05418	 Recall: 0.17533	NDCG: 0.15038
The time for epoch 370 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 58.19574, F1-score: 0.082888 	 Precision: 0.05426	 Recall: 0.17546	NDCG: 0.15051
The time for epoch 371 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.71839, F1-score: 0.082980 	 Precision: 0.05432	 Recall: 0.17567	NDCG: 0.15064
The time for epoch 372 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.01095, F1-score: 0.082950 	 Precision: 0.05430	 Recall: 0.17565	NDCG: 0.15069
The time for epoch 373 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 61.15639, F1-score: 0.082920 	 Precision: 0.05428	 Recall: 0.17558	NDCG: 0.15077
The time for epoch 374 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.59851, F1-score: 0.083062 	 Precision: 0.05436	 Recall: 0.17599	NDCG: 0.15093
The time for epoch 375 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 60.83950, F1-score: 0.083041 	 Precision: 0.05434	 Recall: 0.17599	NDCG: 0.15090
The time for epoch 376 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.14447, F1-score: 0.083154 	 Precision: 0.05441	 Recall: 0.17623	NDCG: 0.15102
The time for epoch 377 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 60.79761, F1-score: 0.083078 	 Precision: 0.05436	 Recall: 0.17610	NDCG: 0.15096
The time for epoch 378 is: train time = 00: 00: 51, test time = 00: 00: 07
Loss = 58.78098, F1-score: 0.083154 	 Precision: 0.05441	 Recall: 0.17624	NDCG: 0.15107
The time for epoch 379 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 58.22987, F1-score: 0.083235 	 Precision: 0.05446	 Recall: 0.17647	NDCG: 0.15120
The time for epoch 380 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 59.09694, F1-score: 0.083226 	 Precision: 0.05446	 Recall: 0.17643	NDCG: 0.15130
The time for epoch 381 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.46354, F1-score: 0.083363 	 Precision: 0.05455	 Recall: 0.17672	NDCG: 0.15138
The time for epoch 382 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.02407, F1-score: 0.083287 	 Precision: 0.05450	 Recall: 0.17653	NDCG: 0.15132
The time for epoch 383 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 61.36520, F1-score: 0.083390 	 Precision: 0.05457	 Recall: 0.17669	NDCG: 0.15152
The time for epoch 384 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.06241, F1-score: 0.083397 	 Precision: 0.05457	 Recall: 0.17680	NDCG: 0.15155
The time for epoch 385 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 59.41306, F1-score: 0.083484 	 Precision: 0.05462	 Recall: 0.17700	NDCG: 0.15167
The time for epoch 386 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.64622, F1-score: 0.083637 	 Precision: 0.05472	 Recall: 0.17734	NDCG: 0.15176
The time for epoch 387 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.87531, F1-score: 0.083600 	 Precision: 0.05470	 Recall: 0.17727	NDCG: 0.15177
The time for epoch 388 is: train time = 00: 01: 02, test time = 00: 00: 10
Loss = 59.95826, F1-score: 0.083864 	 Precision: 0.05485	 Recall: 0.17804	NDCG: 0.15230
The time for epoch 389 is: train time = 00: 01: 03, test time = 00: 00: 08
Loss = 61.29470, F1-score: 0.083743 	 Precision: 0.05477	 Recall: 0.17780	NDCG: 0.15207
The time for epoch 390 is: train time = 00: 01: 03, test time = 00: 00: 09
Loss = 61.78463, F1-score: 0.083778 	 Precision: 0.05479	 Recall: 0.17787	NDCG: 0.15227
The time for epoch 391 is: train time = 00: 01: 02, test time = 00: 00: 08
Loss = 62.62793, F1-score: 0.083833 	 Precision: 0.05483	 Recall: 0.17801	NDCG: 0.15229
The time for epoch 392 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.76062, F1-score: 0.083962 	 Precision: 0.05492	 Recall: 0.17819	NDCG: 0.15242
The time for epoch 393 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.27732, F1-score: 0.083956 	 Precision: 0.05493	 Recall: 0.17805	NDCG: 0.15231
The time for epoch 394 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.91683, F1-score: 0.083959 	 Precision: 0.05492	 Recall: 0.17811	NDCG: 0.15248
The time for epoch 395 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.39681, F1-score: 0.084007 	 Precision: 0.05494	 Recall: 0.17835	NDCG: 0.15251
The time for epoch 396 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 60.47360, F1-score: 0.084089 	 Precision: 0.05500	 Recall: 0.17850	NDCG: 0.15282
The time for epoch 397 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 60.69608, F1-score: 0.084192 	 Precision: 0.05506	 Recall: 0.17880	NDCG: 0.15282
The time for epoch 398 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.61620, F1-score: 0.084181 	 Precision: 0.05505	 Recall: 0.17879	NDCG: 0.15287
The time for epoch 399 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 60.65810, F1-score: 0.084279 	 Precision: 0.05511	 Recall: 0.17900	NDCG: 0.15303
The time for epoch 400 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.43064, F1-score: 0.084199 	 Precision: 0.05507	 Recall: 0.17872	NDCG: 0.15286
The time for epoch 401 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 60.82668, F1-score: 0.084222 	 Precision: 0.05508	 Recall: 0.17884	NDCG: 0.15300
The time for epoch 402 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.43839, F1-score: 0.084253 	 Precision: 0.05511	 Recall: 0.17886	NDCG: 0.15302
The time for epoch 403 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.97804, F1-score: 0.084312 	 Precision: 0.05514	 Recall: 0.17903	NDCG: 0.15313
The time for epoch 404 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.01682, F1-score: 0.084361 	 Precision: 0.05518	 Recall: 0.17907	NDCG: 0.15311
The time for epoch 405 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.20882, F1-score: 0.084517 	 Precision: 0.05527	 Recall: 0.17954	NDCG: 0.15344
The time for epoch 406 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.04555, F1-score: 0.084401 	 Precision: 0.05519	 Recall: 0.17931	NDCG: 0.15336
The time for epoch 407 is: train time = 00: 00: 55, test time = 00: 00: 08
Loss = 62.32212, F1-score: 0.084570 	 Precision: 0.05530	 Recall: 0.17965	NDCG: 0.15360
The time for epoch 408 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.55248, F1-score: 0.084556 	 Precision: 0.05529	 Recall: 0.17961	NDCG: 0.15365
The time for epoch 409 is: train time = 00: 00: 59, test time = 00: 00: 12
Loss = 61.08609, F1-score: 0.084579 	 Precision: 0.05530	 Recall: 0.17971	NDCG: 0.15356
The time for epoch 410 is: train time = 00: 01: 10, test time = 00: 00: 08
Loss = 60.20558, F1-score: 0.084413 	 Precision: 0.05519	 Recall: 0.17943	NDCG: 0.15340
The time for epoch 411 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.13844, F1-score: 0.084511 	 Precision: 0.05526	 Recall: 0.17960	NDCG: 0.15364
The time for epoch 412 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.86485, F1-score: 0.084596 	 Precision: 0.05531	 Recall: 0.17980	NDCG: 0.15379
The time for epoch 413 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.67273, F1-score: 0.084594 	 Precision: 0.05532	 Recall: 0.17972	NDCG: 0.15377
The time for epoch 414 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.85046, F1-score: 0.084758 	 Precision: 0.05542	 Recall: 0.18007	NDCG: 0.15391
The time for epoch 415 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.42414, F1-score: 0.084761 	 Precision: 0.05542	 Recall: 0.18015	NDCG: 0.15405
The time for epoch 416 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.72744, F1-score: 0.084675 	 Precision: 0.05536	 Recall: 0.17999	NDCG: 0.15393
The time for epoch 417 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.28543, F1-score: 0.084758 	 Precision: 0.05540	 Recall: 0.18031	NDCG: 0.15417
The time for epoch 418 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.37897, F1-score: 0.084781 	 Precision: 0.05543	 Recall: 0.18019	NDCG: 0.15419
The time for epoch 419 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.17794, F1-score: 0.084813 	 Precision: 0.05544	 Recall: 0.18041	NDCG: 0.15421
The time for epoch 420 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.56818, F1-score: 0.084878 	 Precision: 0.05549	 Recall: 0.18047	NDCG: 0.15428
The time for epoch 421 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 60.83916, F1-score: 0.084846 	 Precision: 0.05547	 Recall: 0.18034	NDCG: 0.15431
The time for epoch 422 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.30851, F1-score: 0.084922 	 Precision: 0.05552	 Recall: 0.18049	NDCG: 0.15435
The time for epoch 423 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.51269, F1-score: 0.084954 	 Precision: 0.05554	 Recall: 0.18064	NDCG: 0.15446
The time for epoch 424 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.92877, F1-score: 0.084938 	 Precision: 0.05552	 Recall: 0.18063	NDCG: 0.15448
The time for epoch 425 is: train time = 00: 01: 03, test time = 00: 00: 10
Loss = 63.68523, F1-score: 0.084971 	 Precision: 0.05555	 Recall: 0.18061	NDCG: 0.15452
The time for epoch 426 is: train time = 00: 00: 59, test time = 00: 00: 07
Loss = 63.08073, F1-score: 0.085057 	 Precision: 0.05560	 Recall: 0.18088	NDCG: 0.15457
The time for epoch 427 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.11194, F1-score: 0.084886 	 Precision: 0.05548	 Recall: 0.18061	NDCG: 0.15455
The time for epoch 428 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.17870, F1-score: 0.085028 	 Precision: 0.05558	 Recall: 0.18088	NDCG: 0.15463
The time for epoch 429 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 61.02194, F1-score: 0.085083 	 Precision: 0.05562	 Recall: 0.18096	NDCG: 0.15467
The time for epoch 430 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.23428, F1-score: 0.085209 	 Precision: 0.05570	 Recall: 0.18126	NDCG: 0.15487
The time for epoch 431 is: train time = 00: 00: 55, test time = 00: 00: 08
Loss = 61.41409, F1-score: 0.085171 	 Precision: 0.05568	 Recall: 0.18113	NDCG: 0.15481
The time for epoch 432 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 62.55352, F1-score: 0.085215 	 Precision: 0.05570	 Recall: 0.18129	NDCG: 0.15482
The time for epoch 433 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.45049, F1-score: 0.085321 	 Precision: 0.05576	 Recall: 0.18155	NDCG: 0.15503
The time for epoch 434 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.80026, F1-score: 0.085255 	 Precision: 0.05572	 Recall: 0.18143	NDCG: 0.15496
The time for epoch 435 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.06363, F1-score: 0.085311 	 Precision: 0.05576	 Recall: 0.18153	NDCG: 0.15499
The time for epoch 436 is: train time = 00: 00: 52, test time = 00: 00: 09
Loss = 61.18881, F1-score: 0.085288 	 Precision: 0.05574	 Recall: 0.18148	NDCG: 0.15489
The time for epoch 437 is: train time = 00: 01: 08, test time = 00: 00: 10
Loss = 62.02407, F1-score: 0.085352 	 Precision: 0.05578	 Recall: 0.18162	NDCG: 0.15505
The time for epoch 438 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 60.04904, F1-score: 0.085324 	 Precision: 0.05577	 Recall: 0.18154	NDCG: 0.15502
The time for epoch 439 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 61.93344, F1-score: 0.085437 	 Precision: 0.05583	 Recall: 0.18185	NDCG: 0.15518
The time for epoch 440 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.40543, F1-score: 0.085476 	 Precision: 0.05587	 Recall: 0.18185	NDCG: 0.15527
The time for epoch 441 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 63.34756, F1-score: 0.085499 	 Precision: 0.05587	 Recall: 0.18202	NDCG: 0.15530
The time for epoch 442 is: train time = 00: 00: 53, test time = 00: 00: 09
Loss = 61.83181, F1-score: 0.085574 	 Precision: 0.05592	 Recall: 0.18218	NDCG: 0.15541
The time for epoch 443 is: train time = 00: 01: 07, test time = 00: 00: 09
Loss = 61.75662, F1-score: 0.085531 	 Precision: 0.05589	 Recall: 0.18212	NDCG: 0.15535
The time for epoch 444 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.82692, F1-score: 0.085581 	 Precision: 0.05593	 Recall: 0.18218	NDCG: 0.15559
The time for epoch 445 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.32314, F1-score: 0.085571 	 Precision: 0.05592	 Recall: 0.18218	NDCG: 0.15558
The time for epoch 446 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 61.05498, F1-score: 0.085619 	 Precision: 0.05596	 Recall: 0.18218	NDCG: 0.15551
The time for epoch 447 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 62.47181, F1-score: 0.085555 	 Precision: 0.05592	 Recall: 0.18203	NDCG: 0.15553
The time for epoch 448 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.24133, F1-score: 0.085563 	 Precision: 0.05592	 Recall: 0.18213	NDCG: 0.15556
The time for epoch 449 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.63581, F1-score: 0.085679 	 Precision: 0.05599	 Recall: 0.18238	NDCG: 0.15557
The time for epoch 450 is: train time = 00: 01: 07, test time = 00: 00: 09
Loss = 63.50594, F1-score: 0.085757 	 Precision: 0.05604	 Recall: 0.18262	NDCG: 0.15592
The time for epoch 451 is: train time = 00: 00: 56, test time = 00: 00: 08
Loss = 62.01239, F1-score: 0.085748 	 Precision: 0.05603	 Recall: 0.18259	NDCG: 0.15589
The time for epoch 452 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.70237, F1-score: 0.085784 	 Precision: 0.05606	 Recall: 0.18262	NDCG: 0.15592
The time for epoch 453 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.42231, F1-score: 0.085766 	 Precision: 0.05604	 Recall: 0.18268	NDCG: 0.15597
The time for epoch 454 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 62.87130, F1-score: 0.085833 	 Precision: 0.05608	 Recall: 0.18289	NDCG: 0.15602
The time for epoch 455 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.86706, F1-score: 0.085891 	 Precision: 0.05612	 Recall: 0.18290	NDCG: 0.15610
The time for epoch 456 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.40931, F1-score: 0.085841 	 Precision: 0.05608	 Recall: 0.18291	NDCG: 0.15612
The time for epoch 457 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 62.42078, F1-score: 0.085880 	 Precision: 0.05611	 Recall: 0.18292	NDCG: 0.15608
The time for epoch 458 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.76728, F1-score: 0.085940 	 Precision: 0.05615	 Recall: 0.18311	NDCG: 0.15616
The time for epoch 459 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.24226, F1-score: 0.085919 	 Precision: 0.05615	 Recall: 0.18285	NDCG: 0.15609
The time for epoch 460 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.46173, F1-score: 0.086070 	 Precision: 0.05624	 Recall: 0.18333	NDCG: 0.15617
The time for epoch 461 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.50342, F1-score: 0.086042 	 Precision: 0.05621	 Recall: 0.18332	NDCG: 0.15641
The time for epoch 462 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 62.66107, F1-score: 0.086034 	 Precision: 0.05622	 Recall: 0.18318	NDCG: 0.15631
The time for epoch 463 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.10093, F1-score: 0.086033 	 Precision: 0.05620	 Recall: 0.18335	NDCG: 0.15629
The time for epoch 464 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.62811, F1-score: 0.085995 	 Precision: 0.05619	 Recall: 0.18318	NDCG: 0.15628
The time for epoch 465 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.68230, F1-score: 0.086023 	 Precision: 0.05620	 Recall: 0.18333	NDCG: 0.15642
The time for epoch 466 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 62.26539, F1-score: 0.086113 	 Precision: 0.05626	 Recall: 0.18341	NDCG: 0.15646
The time for epoch 467 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.19152, F1-score: 0.086197 	 Precision: 0.05632	 Recall: 0.18361	NDCG: 0.15669
The time for epoch 468 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.64526, F1-score: 0.086103 	 Precision: 0.05626	 Recall: 0.18336	NDCG: 0.15640
The time for epoch 469 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.90230, F1-score: 0.086110 	 Precision: 0.05626	 Recall: 0.18338	NDCG: 0.15649
The time for epoch 470 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 62.69162, F1-score: 0.086108 	 Precision: 0.05626	 Recall: 0.18346	NDCG: 0.15643
The time for epoch 471 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.10974, F1-score: 0.086182 	 Precision: 0.05630	 Recall: 0.18371	NDCG: 0.15655
The time for epoch 472 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.82450, F1-score: 0.086255 	 Precision: 0.05635	 Recall: 0.18379	NDCG: 0.15670
The time for epoch 473 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 62.42799, F1-score: 0.086317 	 Precision: 0.05640	 Recall: 0.18388	NDCG: 0.15671
The time for epoch 474 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 61.96224, F1-score: 0.086271 	 Precision: 0.05637	 Recall: 0.18378	NDCG: 0.15670
The time for epoch 475 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 62.31679, F1-score: 0.086339 	 Precision: 0.05640	 Recall: 0.18399	NDCG: 0.15674
The time for epoch 476 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.35176, F1-score: 0.086264 	 Precision: 0.05635	 Recall: 0.18389	NDCG: 0.15664
The time for epoch 477 is: train time = 00: 00: 48, test time = 00: 00: 08
Loss = 62.98798, F1-score: 0.086262 	 Precision: 0.05636	 Recall: 0.18375	NDCG: 0.15668
The time for epoch 478 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.10463, F1-score: 0.086289 	 Precision: 0.05637	 Recall: 0.18391	NDCG: 0.15678
The time for epoch 479 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 63.12866, F1-score: 0.086230 	 Precision: 0.05635	 Recall: 0.18362	NDCG: 0.15662
The time for epoch 480 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 61.35327, F1-score: 0.086263 	 Precision: 0.05636	 Recall: 0.18376	NDCG: 0.15678
The time for epoch 481 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.43793, F1-score: 0.086325 	 Precision: 0.05640	 Recall: 0.18391	NDCG: 0.15673
The time for epoch 482 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.29696, F1-score: 0.086279 	 Precision: 0.05637	 Recall: 0.18378	NDCG: 0.15671
The time for epoch 483 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.25647, F1-score: 0.086462 	 Precision: 0.05649	 Recall: 0.18420	NDCG: 0.15692
The time for epoch 484 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.41501, F1-score: 0.086408 	 Precision: 0.05645	 Recall: 0.18417	NDCG: 0.15675
The time for epoch 485 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.23597, F1-score: 0.086408 	 Precision: 0.05645	 Recall: 0.18417	NDCG: 0.15686
The time for epoch 486 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.63719, F1-score: 0.086378 	 Precision: 0.05644	 Recall: 0.18393	NDCG: 0.15693
The time for epoch 487 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 62.51512, F1-score: 0.086459 	 Precision: 0.05649	 Recall: 0.18420	NDCG: 0.15688
The time for epoch 488 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.56340, F1-score: 0.086383 	 Precision: 0.05645	 Recall: 0.18391	NDCG: 0.15680
The time for epoch 489 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.01114, F1-score: 0.086475 	 Precision: 0.05650	 Recall: 0.18424	NDCG: 0.15688
The time for epoch 490 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.26086, F1-score: 0.086445 	 Precision: 0.05648	 Recall: 0.18416	NDCG: 0.15691
The time for epoch 491 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.62111, F1-score: 0.086472 	 Precision: 0.05650	 Recall: 0.18414	NDCG: 0.15686
The time for epoch 492 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 63.82639, F1-score: 0.086525 	 Precision: 0.05653	 Recall: 0.18432	NDCG: 0.15687
The time for epoch 493 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.98880, F1-score: 0.086451 	 Precision: 0.05649	 Recall: 0.18412	NDCG: 0.15680
The time for epoch 494 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.04935, F1-score: 0.086440 	 Precision: 0.05648	 Recall: 0.18408	NDCG: 0.15688
The time for epoch 495 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 63.52432, F1-score: 0.086498 	 Precision: 0.05653	 Recall: 0.18412	NDCG: 0.15700
The time for epoch 496 is: train time = 00: 00: 47, test time = 00: 00: 08
Loss = 62.51726, F1-score: 0.086554 	 Precision: 0.05656	 Recall: 0.18425	NDCG: 0.15703
The time for epoch 497 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.72973, F1-score: 0.086506 	 Precision: 0.05652	 Recall: 0.18422	NDCG: 0.15708
The time for epoch 498 is: train time = 00: 00: 48, test time = 00: 00: 08
Loss = 63.84245, F1-score: 0.086574 	 Precision: 0.05657	 Recall: 0.18432	NDCG: 0.15709
The time for epoch 499 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 62.23698, F1-score: 0.086631 	 Precision: 0.05660	 Recall: 0.18458	NDCG: 0.15726
The time for epoch 500 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.51423, F1-score: 0.086665 	 Precision: 0.05662	 Recall: 0.18467	NDCG: 0.15719
The time for epoch 501 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 62.55478, F1-score: 0.086560 	 Precision: 0.05655	 Recall: 0.18439	NDCG: 0.15702
The time for epoch 502 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.30302, F1-score: 0.086541 	 Precision: 0.05654	 Recall: 0.18433	NDCG: 0.15697
The time for epoch 503 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 61.74058, F1-score: 0.086629 	 Precision: 0.05660	 Recall: 0.18449	NDCG: 0.15718
The time for epoch 504 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.41238, F1-score: 0.086607 	 Precision: 0.05658	 Recall: 0.18450	NDCG: 0.15711
The time for epoch 505 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 63.10782, F1-score: 0.086599 	 Precision: 0.05657	 Recall: 0.18454	NDCG: 0.15704
The time for epoch 506 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 62.98642, F1-score: 0.086684 	 Precision: 0.05663	 Recall: 0.18471	NDCG: 0.15727
The time for epoch 507 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.35202, F1-score: 0.086580 	 Precision: 0.05657	 Recall: 0.18443	NDCG: 0.15706
The time for epoch 508 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.45529, F1-score: 0.086750 	 Precision: 0.05668	 Recall: 0.18478	NDCG: 0.15723
The time for epoch 509 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.31490, F1-score: 0.086663 	 Precision: 0.05662	 Recall: 0.18465	NDCG: 0.15721
The time for epoch 510 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 63.92182, F1-score: 0.086739 	 Precision: 0.05667	 Recall: 0.18479	NDCG: 0.15729
The time for epoch 511 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.57886, F1-score: 0.086780 	 Precision: 0.05670	 Recall: 0.18484	NDCG: 0.15732
The time for epoch 512 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.79658, F1-score: 0.086775 	 Precision: 0.05670	 Recall: 0.18481	NDCG: 0.15725
The time for epoch 513 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.38838, F1-score: 0.086715 	 Precision: 0.05666	 Recall: 0.18466	NDCG: 0.15724
The time for epoch 514 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.77779, F1-score: 0.086755 	 Precision: 0.05667	 Recall: 0.18488	NDCG: 0.15742
The time for epoch 515 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.46288, F1-score: 0.086765 	 Precision: 0.05668	 Recall: 0.18488	NDCG: 0.15746
The time for epoch 516 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.35911, F1-score: 0.086797 	 Precision: 0.05670	 Recall: 0.18497	NDCG: 0.15731
The time for epoch 517 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.42933, F1-score: 0.086860 	 Precision: 0.05676	 Recall: 0.18493	NDCG: 0.15742
The time for epoch 518 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.21148, F1-score: 0.086874 	 Precision: 0.05676	 Recall: 0.18502	NDCG: 0.15752
The time for epoch 519 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 62.88721, F1-score: 0.086823 	 Precision: 0.05673	 Recall: 0.18488	NDCG: 0.15729
The time for epoch 520 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.37201, F1-score: 0.086860 	 Precision: 0.05676	 Recall: 0.18490	NDCG: 0.15744
The time for epoch 521 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.42640, F1-score: 0.086792 	 Precision: 0.05672	 Recall: 0.18477	NDCG: 0.15733
The time for epoch 522 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.60342, F1-score: 0.086800 	 Precision: 0.05672	 Recall: 0.18482	NDCG: 0.15746
The time for epoch 523 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 62.78617, F1-score: 0.086829 	 Precision: 0.05674	 Recall: 0.18491	NDCG: 0.15745
The time for epoch 524 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.13384, F1-score: 0.086944 	 Precision: 0.05681	 Recall: 0.18518	NDCG: 0.15760
The time for epoch 525 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.51397, F1-score: 0.086832 	 Precision: 0.05674	 Recall: 0.18492	NDCG: 0.15739
The time for epoch 526 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.13932, F1-score: 0.086841 	 Precision: 0.05675	 Recall: 0.18486	NDCG: 0.15748
The time for epoch 527 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 63.01353, F1-score: 0.086858 	 Precision: 0.05676	 Recall: 0.18491	NDCG: 0.15750
The time for epoch 528 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.39720, F1-score: 0.086830 	 Precision: 0.05673	 Recall: 0.18495	NDCG: 0.15743
The time for epoch 529 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.20985, F1-score: 0.086855 	 Precision: 0.05676	 Recall: 0.18488	NDCG: 0.15749
The time for epoch 530 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 62.49307, F1-score: 0.086901 	 Precision: 0.05677	 Recall: 0.18517	NDCG: 0.15753
The time for epoch 531 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.34064, F1-score: 0.086828 	 Precision: 0.05672	 Recall: 0.18504	NDCG: 0.15734
The time for epoch 532 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.47662, F1-score: 0.086847 	 Precision: 0.05674	 Recall: 0.18502	NDCG: 0.15751
The time for epoch 533 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.55931, F1-score: 0.086974 	 Precision: 0.05682	 Recall: 0.18532	NDCG: 0.15763
The time for epoch 534 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.48224, F1-score: 0.086994 	 Precision: 0.05684	 Recall: 0.18534	NDCG: 0.15758
The time for epoch 535 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 63.45317, F1-score: 0.086980 	 Precision: 0.05683	 Recall: 0.18526	NDCG: 0.15752
The time for epoch 536 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.78088, F1-score: 0.086947 	 Precision: 0.05682	 Recall: 0.18507	NDCG: 0.15761
The time for epoch 537 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.57882, F1-score: 0.086989 	 Precision: 0.05684	 Recall: 0.18524	NDCG: 0.15756
The time for epoch 538 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.16907, F1-score: 0.086962 	 Precision: 0.05682	 Recall: 0.18522	NDCG: 0.15771
The time for epoch 539 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.00088, F1-score: 0.086779 	 Precision: 0.05671	 Recall: 0.18476	NDCG: 0.15745
The time for epoch 540 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 65.51475, F1-score: 0.086910 	 Precision: 0.05680	 Recall: 0.18497	NDCG: 0.15751
The time for epoch 541 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.95099, F1-score: 0.087031 	 Precision: 0.05686	 Recall: 0.18539	NDCG: 0.15773
The time for epoch 542 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 65.25154, F1-score: 0.086933 	 Precision: 0.05680	 Recall: 0.18517	NDCG: 0.15759
The time for epoch 543 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 65.05615, F1-score: 0.086868 	 Precision: 0.05676	 Recall: 0.18503	NDCG: 0.15757
The time for epoch 544 is: train time = 00: 00: 58, test time = 00: 00: 10
Loss = 63.76131, F1-score: 0.086974 	 Precision: 0.05682	 Recall: 0.18534	NDCG: 0.15763
The time for epoch 545 is: train time = 00: 01: 04, test time = 00: 00: 08
Loss = 62.69662, F1-score: 0.087052 	 Precision: 0.05689	 Recall: 0.18533	NDCG: 0.15768
The time for epoch 546 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.84192, F1-score: 0.087034 	 Precision: 0.05687	 Recall: 0.18536	NDCG: 0.15769
The time for epoch 547 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.05976, F1-score: 0.087045 	 Precision: 0.05688	 Recall: 0.18532	NDCG: 0.15771
The time for epoch 548 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 63.30070, F1-score: 0.086970 	 Precision: 0.05682	 Recall: 0.18526	NDCG: 0.15764
The time for epoch 549 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.12058, F1-score: 0.087028 	 Precision: 0.05686	 Recall: 0.18536	NDCG: 0.15777
The time for epoch 550 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.94517, F1-score: 0.087007 	 Precision: 0.05684	 Recall: 0.18544	NDCG: 0.15774
The time for epoch 551 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.91959, F1-score: 0.086932 	 Precision: 0.05680	 Recall: 0.18517	NDCG: 0.15757
The time for epoch 552 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 63.70369, F1-score: 0.087031 	 Precision: 0.05686	 Recall: 0.18544	NDCG: 0.15768
The time for epoch 553 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 63.96640, F1-score: 0.087097 	 Precision: 0.05691	 Recall: 0.18546	NDCG: 0.15770
The time for epoch 554 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.73144, F1-score: 0.087134 	 Precision: 0.05693	 Recall: 0.18563	NDCG: 0.15772
The time for epoch 555 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 65.10108, F1-score: 0.087131 	 Precision: 0.05693	 Recall: 0.18562	NDCG: 0.15771
The time for epoch 556 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 63.69295, F1-score: 0.087060 	 Precision: 0.05688	 Recall: 0.18542	NDCG: 0.15770
The time for epoch 557 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.01680, F1-score: 0.087091 	 Precision: 0.05690	 Recall: 0.18551	NDCG: 0.15778
The time for epoch 558 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.72842, F1-score: 0.087077 	 Precision: 0.05688	 Recall: 0.18560	NDCG: 0.15767
The time for epoch 559 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.82471, F1-score: 0.087071 	 Precision: 0.05688	 Recall: 0.18561	NDCG: 0.15765
The time for epoch 560 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 65.07771, F1-score: 0.087057 	 Precision: 0.05687	 Recall: 0.18553	NDCG: 0.15762
The time for epoch 561 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.10477, F1-score: 0.087134 	 Precision: 0.05692	 Recall: 0.18572	NDCG: 0.15766
The time for epoch 562 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.92939, F1-score: 0.087091 	 Precision: 0.05690	 Recall: 0.18558	NDCG: 0.15775
The time for epoch 563 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.49266, F1-score: 0.087014 	 Precision: 0.05684	 Recall: 0.18543	NDCG: 0.15764
The time for epoch 564 is: train time = 00: 00: 52, test time = 00: 00: 09
Loss = 63.17026, F1-score: 0.087178 	 Precision: 0.05695	 Recall: 0.18578	NDCG: 0.15780
The time for epoch 565 is: train time = 00: 01: 04, test time = 00: 00: 10
Loss = 64.48538, F1-score: 0.087072 	 Precision: 0.05688	 Recall: 0.18553	NDCG: 0.15772
The time for epoch 566 is: train time = 00: 00: 52, test time = 00: 00: 07
Loss = 65.22326, F1-score: 0.087111 	 Precision: 0.05692	 Recall: 0.18554	NDCG: 0.15776
The time for epoch 567 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.31039, F1-score: 0.087124 	 Precision: 0.05692	 Recall: 0.18563	NDCG: 0.15776
The time for epoch 568 is: train time = 00: 00: 49, test time = 00: 00: 08
Loss = 64.35140, F1-score: 0.087215 	 Precision: 0.05698	 Recall: 0.18585	NDCG: 0.15776
The time for epoch 569 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 62.19867, F1-score: 0.087182 	 Precision: 0.05696	 Recall: 0.18573	NDCG: 0.15780
The time for epoch 570 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.68198, F1-score: 0.087187 	 Precision: 0.05696	 Recall: 0.18576	NDCG: 0.15777
The time for epoch 571 is: train time = 00: 00: 48, test time = 00: 00: 08
Loss = 63.18546, F1-score: 0.087144 	 Precision: 0.05693	 Recall: 0.18567	NDCG: 0.15774
The time for epoch 572 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.27798, F1-score: 0.087085 	 Precision: 0.05689	 Recall: 0.18563	NDCG: 0.15787
The time for epoch 573 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 65.38540, F1-score: 0.087089 	 Precision: 0.05689	 Recall: 0.18563	NDCG: 0.15770
The time for epoch 574 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.64009, F1-score: 0.087073 	 Precision: 0.05689	 Recall: 0.18552	NDCG: 0.15770
The time for epoch 575 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.19482, F1-score: 0.087153 	 Precision: 0.05694	 Recall: 0.18568	NDCG: 0.15769
The time for epoch 576 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 65.65012, F1-score: 0.087183 	 Precision: 0.05696	 Recall: 0.18575	NDCG: 0.15772
The time for epoch 577 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.13501, F1-score: 0.087102 	 Precision: 0.05691	 Recall: 0.18555	NDCG: 0.15761
The time for epoch 578 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.76894, F1-score: 0.087082 	 Precision: 0.05689	 Recall: 0.18558	NDCG: 0.15755
The time for epoch 579 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.75553, F1-score: 0.087055 	 Precision: 0.05687	 Recall: 0.18548	NDCG: 0.15756
The time for epoch 580 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.26263, F1-score: 0.087139 	 Precision: 0.05693	 Recall: 0.18562	NDCG: 0.15785
The time for epoch 581 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.39335, F1-score: 0.087159 	 Precision: 0.05695	 Recall: 0.18560	NDCG: 0.15775
The time for epoch 582 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 65.94884, F1-score: 0.087151 	 Precision: 0.05694	 Recall: 0.18566	NDCG: 0.15777
The time for epoch 583 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.65685, F1-score: 0.087118 	 Precision: 0.05692	 Recall: 0.18557	NDCG: 0.15773
The time for epoch 584 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.61295, F1-score: 0.087086 	 Precision: 0.05690	 Recall: 0.18550	NDCG: 0.15759
The time for epoch 585 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.52079, F1-score: 0.087094 	 Precision: 0.05691	 Recall: 0.18550	NDCG: 0.15780
The time for epoch 586 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.69553, F1-score: 0.087055 	 Precision: 0.05688	 Recall: 0.18541	NDCG: 0.15762
The time for epoch 587 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.97204, F1-score: 0.087131 	 Precision: 0.05693	 Recall: 0.18562	NDCG: 0.15776
The time for epoch 588 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.47528, F1-score: 0.087127 	 Precision: 0.05692	 Recall: 0.18560	NDCG: 0.15763
The time for epoch 589 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.09975, F1-score: 0.087212 	 Precision: 0.05698	 Recall: 0.18582	NDCG: 0.15777
The time for epoch 590 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.27277, F1-score: 0.087199 	 Precision: 0.05696	 Recall: 0.18587	NDCG: 0.15780
The time for epoch 591 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 65.19199, F1-score: 0.087214 	 Precision: 0.05698	 Recall: 0.18584	NDCG: 0.15784
The time for epoch 592 is: train time = 00: 00: 48, test time = 00: 00: 08
Loss = 63.49570, F1-score: 0.087253 	 Precision: 0.05701	 Recall: 0.18587	NDCG: 0.15779
The time for epoch 593 is: train time = 00: 00: 48, test time = 00: 00: 08
Loss = 64.57928, F1-score: 0.087177 	 Precision: 0.05696	 Recall: 0.18570	NDCG: 0.15782
The time for epoch 594 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 65.03943, F1-score: 0.087272 	 Precision: 0.05702	 Recall: 0.18590	NDCG: 0.15785
The time for epoch 595 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.71958, F1-score: 0.087268 	 Precision: 0.05701	 Recall: 0.18595	NDCG: 0.15799
The time for epoch 596 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 65.46424, F1-score: 0.087354 	 Precision: 0.05707	 Recall: 0.18615	NDCG: 0.15805
The time for epoch 597 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 64.65134, F1-score: 0.087321 	 Precision: 0.05705	 Recall: 0.18600	NDCG: 0.15794
The time for epoch 598 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 63.45425, F1-score: 0.087283 	 Precision: 0.05703	 Recall: 0.18588	NDCG: 0.15806
The time for epoch 599 is: train time = 00: 00: 50, test time = 00: 00: 08
Loss = 64.62415, F1-score: 0.087311 	 Precision: 0.05705	 Recall: 0.18599	NDCG: 0.15800
The time for epoch 600 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 65.21112, F1-score: 0.087324 	 Precision: 0.05706	 Recall: 0.18600	NDCG: 0.15800
The time for epoch 601 is: train time = 00: 00: 51, test time = 00: 00: 08
Loss = 63.99644, F1-score: 0.087266 	 Precision: 0.05701	 Recall: 0.18594	NDCG: 0.15798
The time for epoch 602 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.03024, F1-score: 0.087301 	 Precision: 0.05703	 Recall: 0.18605	NDCG: 0.15799
The time for epoch 603 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.64330, F1-score: 0.087246 	 Precision: 0.05699	 Recall: 0.18597	NDCG: 0.15788
The time for epoch 604 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.63600, F1-score: 0.087205 	 Precision: 0.05697	 Recall: 0.18580	NDCG: 0.15786
The time for epoch 605 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.87014, F1-score: 0.087211 	 Precision: 0.05698	 Recall: 0.18581	NDCG: 0.15795
The time for epoch 606 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.01654, F1-score: 0.087267 	 Precision: 0.05702	 Recall: 0.18589	NDCG: 0.15795
The time for epoch 607 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.16801, F1-score: 0.087249 	 Precision: 0.05699	 Recall: 0.18596	NDCG: 0.15789
The time for epoch 608 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.48417, F1-score: 0.087211 	 Precision: 0.05698	 Recall: 0.18582	NDCG: 0.15786
The time for epoch 609 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.03622, F1-score: 0.087268 	 Precision: 0.05701	 Recall: 0.18594	NDCG: 0.15788
The time for epoch 610 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.46297, F1-score: 0.087184 	 Precision: 0.05696	 Recall: 0.18575	NDCG: 0.15781
The time for epoch 611 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 63.77692, F1-score: 0.087143 	 Precision: 0.05692	 Recall: 0.18575	NDCG: 0.15770
The time for epoch 612 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.27548, F1-score: 0.087207 	 Precision: 0.05698	 Recall: 0.18569	NDCG: 0.15770
The time for epoch 613 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.21220, F1-score: 0.087258 	 Precision: 0.05701	 Recall: 0.18592	NDCG: 0.15794
The time for epoch 614 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.02213, F1-score: 0.087305 	 Precision: 0.05703	 Recall: 0.18606	NDCG: 0.15793
The time for epoch 615 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.27043, F1-score: 0.087261 	 Precision: 0.05700	 Recall: 0.18602	NDCG: 0.15799
The time for epoch 616 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.51447, F1-score: 0.087245 	 Precision: 0.05699	 Recall: 0.18596	NDCG: 0.15786
The time for epoch 617 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.60722, F1-score: 0.087272 	 Precision: 0.05701	 Recall: 0.18605	NDCG: 0.15794
The time for epoch 618 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.04782, F1-score: 0.087307 	 Precision: 0.05704	 Recall: 0.18600	NDCG: 0.15798
The time for epoch 619 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.29810, F1-score: 0.087171 	 Precision: 0.05695	 Recall: 0.18574	NDCG: 0.15799
The time for epoch 620 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.70863, F1-score: 0.087148 	 Precision: 0.05693	 Recall: 0.18572	NDCG: 0.15796
The time for epoch 621 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.96656, F1-score: 0.087225 	 Precision: 0.05698	 Recall: 0.18590	NDCG: 0.15784
The time for epoch 622 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.56142, F1-score: 0.087143 	 Precision: 0.05693	 Recall: 0.18571	NDCG: 0.15784
The time for epoch 623 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.83263, F1-score: 0.087234 	 Precision: 0.05698	 Recall: 0.18596	NDCG: 0.15794
The time for epoch 624 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.91983, F1-score: 0.087229 	 Precision: 0.05699	 Recall: 0.18583	NDCG: 0.15779
The time for epoch 625 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.98318, F1-score: 0.087276 	 Precision: 0.05701	 Recall: 0.18607	NDCG: 0.15798
The time for epoch 626 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.41376, F1-score: 0.087303 	 Precision: 0.05704	 Recall: 0.18599	NDCG: 0.15792
The time for epoch 627 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.41306, F1-score: 0.087260 	 Precision: 0.05702	 Recall: 0.18582	NDCG: 0.15790
The time for epoch 628 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 63.33140, F1-score: 0.087199 	 Precision: 0.05697	 Recall: 0.18581	NDCG: 0.15775
The time for epoch 629 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.09308, F1-score: 0.087169 	 Precision: 0.05695	 Recall: 0.18574	NDCG: 0.15781
The time for epoch 630 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.89029, F1-score: 0.087298 	 Precision: 0.05702	 Recall: 0.18609	NDCG: 0.15808
The time for epoch 631 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.59330, F1-score: 0.087287 	 Precision: 0.05702	 Recall: 0.18604	NDCG: 0.15799
The time for epoch 632 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 63.58415, F1-score: 0.087327 	 Precision: 0.05704	 Recall: 0.18615	NDCG: 0.15810
The time for epoch 633 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.76958, F1-score: 0.087329 	 Precision: 0.05703	 Recall: 0.18628	NDCG: 0.15816
The time for epoch 634 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.32970, F1-score: 0.087368 	 Precision: 0.05706	 Recall: 0.18636	NDCG: 0.15810
The time for epoch 635 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.70527, F1-score: 0.087305 	 Precision: 0.05703	 Recall: 0.18613	NDCG: 0.15797
The time for epoch 636 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.50150, F1-score: 0.087164 	 Precision: 0.05694	 Recall: 0.18578	NDCG: 0.15784
The time for epoch 637 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.31088, F1-score: 0.087276 	 Precision: 0.05701	 Recall: 0.18605	NDCG: 0.15787
The time for epoch 638 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 63.73108, F1-score: 0.087278 	 Precision: 0.05701	 Recall: 0.18605	NDCG: 0.15797
The time for epoch 639 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.20873, F1-score: 0.087219 	 Precision: 0.05698	 Recall: 0.18589	NDCG: 0.15787
The time for epoch 640 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.72932, F1-score: 0.087247 	 Precision: 0.05699	 Recall: 0.18596	NDCG: 0.15787
The time for epoch 641 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.41973, F1-score: 0.087247 	 Precision: 0.05700	 Recall: 0.18591	NDCG: 0.15793
The time for epoch 642 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.56689, F1-score: 0.087227 	 Precision: 0.05698	 Recall: 0.18588	NDCG: 0.15796
The time for epoch 643 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.03598, F1-score: 0.087302 	 Precision: 0.05703	 Recall: 0.18603	NDCG: 0.15789
The time for epoch 644 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.87768, F1-score: 0.087217 	 Precision: 0.05698	 Recall: 0.18580	NDCG: 0.15799
The time for epoch 645 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.64099, F1-score: 0.087195 	 Precision: 0.05697	 Recall: 0.18576	NDCG: 0.15786
The time for epoch 646 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.22089, F1-score: 0.087171 	 Precision: 0.05695	 Recall: 0.18576	NDCG: 0.15775
The time for epoch 647 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.78604, F1-score: 0.087195 	 Precision: 0.05696	 Recall: 0.18581	NDCG: 0.15780
The time for epoch 648 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 63.82201, F1-score: 0.087295 	 Precision: 0.05702	 Recall: 0.18606	NDCG: 0.15796
The time for epoch 649 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.65131, F1-score: 0.087284 	 Precision: 0.05702	 Recall: 0.18599	NDCG: 0.15797
The time for epoch 650 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.82185, F1-score: 0.087263 	 Precision: 0.05700	 Recall: 0.18598	NDCG: 0.15789
The time for epoch 651 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.44498, F1-score: 0.087236 	 Precision: 0.05699	 Recall: 0.18591	NDCG: 0.15782
The time for epoch 652 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.57862, F1-score: 0.087316 	 Precision: 0.05703	 Recall: 0.18616	NDCG: 0.15792
The time for epoch 653 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 66.10622, F1-score: 0.087280 	 Precision: 0.05701	 Recall: 0.18605	NDCG: 0.15785
The time for epoch 654 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.83862, F1-score: 0.087269 	 Precision: 0.05701	 Recall: 0.18597	NDCG: 0.15780
The time for epoch 655 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.67742, F1-score: 0.087264 	 Precision: 0.05701	 Recall: 0.18596	NDCG: 0.15797
The time for epoch 656 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.72428, F1-score: 0.087390 	 Precision: 0.05709	 Recall: 0.18628	NDCG: 0.15802
The time for epoch 657 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.41863, F1-score: 0.087280 	 Precision: 0.05701	 Recall: 0.18606	NDCG: 0.15799
The time for epoch 658 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.14143, F1-score: 0.087336 	 Precision: 0.05707	 Recall: 0.18600	NDCG: 0.15802
The time for epoch 659 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.17398, F1-score: 0.087368 	 Precision: 0.05708	 Recall: 0.18616	NDCG: 0.15803
The time for epoch 660 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.48585, F1-score: 0.087301 	 Precision: 0.05704	 Recall: 0.18595	NDCG: 0.15802
The time for epoch 661 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.46764, F1-score: 0.087356 	 Precision: 0.05706	 Recall: 0.18623	NDCG: 0.15818
The time for epoch 662 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.99215, F1-score: 0.087177 	 Precision: 0.05694	 Recall: 0.18588	NDCG: 0.15798
The time for epoch 663 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.13857, F1-score: 0.087186 	 Precision: 0.05696	 Recall: 0.18580	NDCG: 0.15785
The time for epoch 664 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.90423, F1-score: 0.087253 	 Precision: 0.05700	 Recall: 0.18593	NDCG: 0.15787
The time for epoch 665 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.54416, F1-score: 0.087268 	 Precision: 0.05701	 Recall: 0.18595	NDCG: 0.15791
The time for epoch 666 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 66.04422, F1-score: 0.087227 	 Precision: 0.05698	 Recall: 0.18587	NDCG: 0.15785
The time for epoch 667 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.02347, F1-score: 0.087277 	 Precision: 0.05701	 Recall: 0.18600	NDCG: 0.15791
The time for epoch 668 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.13517, F1-score: 0.087182 	 Precision: 0.05696	 Recall: 0.18575	NDCG: 0.15781
The time for epoch 669 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.12016, F1-score: 0.087265 	 Precision: 0.05701	 Recall: 0.18591	NDCG: 0.15804
The time for epoch 670 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.35145, F1-score: 0.087248 	 Precision: 0.05699	 Recall: 0.18597	NDCG: 0.15789
The time for epoch 671 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.29092, F1-score: 0.087269 	 Precision: 0.05702	 Recall: 0.18591	NDCG: 0.15792
The time for epoch 672 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.81921, F1-score: 0.087186 	 Precision: 0.05697	 Recall: 0.18566	NDCG: 0.15786
The time for epoch 673 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.89776, F1-score: 0.087245 	 Precision: 0.05700	 Recall: 0.18589	NDCG: 0.15786
The time for epoch 674 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.53574, F1-score: 0.087223 	 Precision: 0.05698	 Recall: 0.18586	NDCG: 0.15780
The time for epoch 675 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.39738, F1-score: 0.087233 	 Precision: 0.05699	 Recall: 0.18587	NDCG: 0.15780
The time for epoch 676 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.00153, F1-score: 0.087209 	 Precision: 0.05697	 Recall: 0.18585	NDCG: 0.15783
The time for epoch 677 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.64207, F1-score: 0.087236 	 Precision: 0.05698	 Recall: 0.18595	NDCG: 0.15783
The time for epoch 678 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.33174, F1-score: 0.087228 	 Precision: 0.05698	 Recall: 0.18591	NDCG: 0.15780
The time for epoch 679 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.52433, F1-score: 0.087175 	 Precision: 0.05694	 Recall: 0.18584	NDCG: 0.15780
The time for epoch 680 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.14756, F1-score: 0.087181 	 Precision: 0.05695	 Recall: 0.18581	NDCG: 0.15774
The time for epoch 681 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.18564, F1-score: 0.087135 	 Precision: 0.05692	 Recall: 0.18576	NDCG: 0.15772
The time for epoch 682 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.88057, F1-score: 0.087265 	 Precision: 0.05700	 Recall: 0.18604	NDCG: 0.15789
The time for epoch 683 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.50599, F1-score: 0.087209 	 Precision: 0.05696	 Recall: 0.18595	NDCG: 0.15787
The time for epoch 684 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.10137, F1-score: 0.087312 	 Precision: 0.05703	 Recall: 0.18614	NDCG: 0.15793
The time for epoch 685 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.51440, F1-score: 0.087271 	 Precision: 0.05701	 Recall: 0.18600	NDCG: 0.15797
The time for epoch 686 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.95487, F1-score: 0.087265 	 Precision: 0.05700	 Recall: 0.18600	NDCG: 0.15792
The time for epoch 687 is: train time = 00: 00: 53, test time = 00: 00: 07
Loss = 65.68890, F1-score: 0.087238 	 Precision: 0.05698	 Recall: 0.18601	NDCG: 0.15795
The time for epoch 688 is: train time = 00: 00: 53, test time = 00: 00: 07
Loss = 64.72176, F1-score: 0.087271 	 Precision: 0.05700	 Recall: 0.18609	NDCG: 0.15788
The time for epoch 689 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.34076, F1-score: 0.087318 	 Precision: 0.05704	 Recall: 0.18614	NDCG: 0.15801
The time for epoch 690 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.62736, F1-score: 0.087316 	 Precision: 0.05704	 Recall: 0.18607	NDCG: 0.15803
The time for epoch 691 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.66693, F1-score: 0.087240 	 Precision: 0.05699	 Recall: 0.18595	NDCG: 0.15780
The time for epoch 692 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.92171, F1-score: 0.087270 	 Precision: 0.05700	 Recall: 0.18610	NDCG: 0.15791
The time for epoch 693 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.04988, F1-score: 0.087276 	 Precision: 0.05700	 Recall: 0.18616	NDCG: 0.15795
The time for epoch 694 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.77653, F1-score: 0.087173 	 Precision: 0.05693	 Recall: 0.18595	NDCG: 0.15791
The time for epoch 695 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.53529, F1-score: 0.087148 	 Precision: 0.05693	 Recall: 0.18576	NDCG: 0.15785
The time for epoch 696 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.53738, F1-score: 0.087182 	 Precision: 0.05695	 Recall: 0.18588	NDCG: 0.15785
The time for epoch 697 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.76983, F1-score: 0.087174 	 Precision: 0.05694	 Recall: 0.18583	NDCG: 0.15780
The time for epoch 698 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.36501, F1-score: 0.087253 	 Precision: 0.05699	 Recall: 0.18600	NDCG: 0.15795
The time for epoch 699 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.74828, F1-score: 0.087282 	 Precision: 0.05701	 Recall: 0.18607	NDCG: 0.15792
The time for epoch 700 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 67.32276, F1-score: 0.087250 	 Precision: 0.05700	 Recall: 0.18592	NDCG: 0.15805
The time for epoch 701 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.12202, F1-score: 0.087178 	 Precision: 0.05696	 Recall: 0.18570	NDCG: 0.15793
The time for epoch 702 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 66.09036, F1-score: 0.087264 	 Precision: 0.05700	 Recall: 0.18599	NDCG: 0.15796
The time for epoch 703 is: train time = 00: 00: 55, test time = 00: 00: 08
Loss = 65.23271, F1-score: 0.087295 	 Precision: 0.05702	 Recall: 0.18610	NDCG: 0.15801
The time for epoch 704 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.80940, F1-score: 0.087281 	 Precision: 0.05701	 Recall: 0.18609	NDCG: 0.15793
The time for epoch 705 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.95245, F1-score: 0.087276 	 Precision: 0.05701	 Recall: 0.18601	NDCG: 0.15791
The time for epoch 706 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 64.91940, F1-score: 0.087307 	 Precision: 0.05703	 Recall: 0.18613	NDCG: 0.15794
The time for epoch 707 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.16480, F1-score: 0.087255 	 Precision: 0.05700	 Recall: 0.18598	NDCG: 0.15791
The time for epoch 708 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.08393, F1-score: 0.087138 	 Precision: 0.05692	 Recall: 0.18577	NDCG: 0.15778
The time for epoch 709 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.23825, F1-score: 0.087227 	 Precision: 0.05697	 Recall: 0.18601	NDCG: 0.15786
The time for epoch 710 is: train time = 00: 00: 53, test time = 00: 00: 07
Loss = 66.41064, F1-score: 0.087332 	 Precision: 0.05705	 Recall: 0.18616	NDCG: 0.15800
The time for epoch 711 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.44118, F1-score: 0.087296 	 Precision: 0.05702	 Recall: 0.18613	NDCG: 0.15795
The time for epoch 712 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 66.05576, F1-score: 0.087162 	 Precision: 0.05694	 Recall: 0.18578	NDCG: 0.15772
The time for epoch 713 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.65654, F1-score: 0.087218 	 Precision: 0.05698	 Recall: 0.18586	NDCG: 0.15790
The time for epoch 714 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.95329, F1-score: 0.087272 	 Precision: 0.05700	 Recall: 0.18613	NDCG: 0.15798
The time for epoch 715 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.91438, F1-score: 0.087334 	 Precision: 0.05704	 Recall: 0.18622	NDCG: 0.15800
The time for epoch 716 is: train time = 00: 00: 52, test time = 00: 00: 07
Loss = 64.82700, F1-score: 0.087323 	 Precision: 0.05703	 Recall: 0.18624	NDCG: 0.15791
The time for epoch 717 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.62588, F1-score: 0.087284 	 Precision: 0.05702	 Recall: 0.18601	NDCG: 0.15803
The time for epoch 718 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.19861, F1-score: 0.087300 	 Precision: 0.05703	 Recall: 0.18605	NDCG: 0.15802
The time for epoch 719 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.48967, F1-score: 0.087338 	 Precision: 0.05705	 Recall: 0.18623	NDCG: 0.15799
The time for epoch 720 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.88163, F1-score: 0.087291 	 Precision: 0.05703	 Recall: 0.18600	NDCG: 0.15785
The time for epoch 721 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.82829, F1-score: 0.087264 	 Precision: 0.05701	 Recall: 0.18597	NDCG: 0.15788
The time for epoch 722 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.02852, F1-score: 0.087293 	 Precision: 0.05702	 Recall: 0.18611	NDCG: 0.15789
The time for epoch 723 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.60016, F1-score: 0.087243 	 Precision: 0.05698	 Recall: 0.18601	NDCG: 0.15788
The time for epoch 724 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.32823, F1-score: 0.087217 	 Precision: 0.05697	 Recall: 0.18589	NDCG: 0.15792
The time for epoch 725 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.50997, F1-score: 0.087355 	 Precision: 0.05706	 Recall: 0.18626	NDCG: 0.15803
The time for epoch 726 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.37051, F1-score: 0.087272 	 Precision: 0.05700	 Recall: 0.18612	NDCG: 0.15788
The time for epoch 727 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.39405, F1-score: 0.087158 	 Precision: 0.05693	 Recall: 0.18585	NDCG: 0.15781
The time for epoch 728 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.46912, F1-score: 0.087219 	 Precision: 0.05697	 Recall: 0.18597	NDCG: 0.15796
The time for epoch 729 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.08350, F1-score: 0.087193 	 Precision: 0.05695	 Recall: 0.18596	NDCG: 0.15799
The time for epoch 730 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.02036, F1-score: 0.087243 	 Precision: 0.05698	 Recall: 0.18607	NDCG: 0.15793
The time for epoch 731 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.35342, F1-score: 0.087288 	 Precision: 0.05701	 Recall: 0.18610	NDCG: 0.15794
The time for epoch 732 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.57536, F1-score: 0.087174 	 Precision: 0.05694	 Recall: 0.18587	NDCG: 0.15787
The time for epoch 733 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.76611, F1-score: 0.087183 	 Precision: 0.05695	 Recall: 0.18588	NDCG: 0.15781
The time for epoch 734 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.45980, F1-score: 0.087181 	 Precision: 0.05694	 Recall: 0.18593	NDCG: 0.15792
The time for epoch 735 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.51279, F1-score: 0.087154 	 Precision: 0.05693	 Recall: 0.18582	NDCG: 0.15785
The time for epoch 736 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.80757, F1-score: 0.087189 	 Precision: 0.05695	 Recall: 0.18585	NDCG: 0.15778
The time for epoch 737 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.05334, F1-score: 0.087187 	 Precision: 0.05694	 Recall: 0.18594	NDCG: 0.15784
The time for epoch 738 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.13953, F1-score: 0.087148 	 Precision: 0.05692	 Recall: 0.18583	NDCG: 0.15791
The time for epoch 739 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.77383, F1-score: 0.087172 	 Precision: 0.05694	 Recall: 0.18580	NDCG: 0.15791
The time for epoch 740 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 64.97133, F1-score: 0.087188 	 Precision: 0.05696	 Recall: 0.18582	NDCG: 0.15787
The time for epoch 741 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.74419, F1-score: 0.087138 	 Precision: 0.05692	 Recall: 0.18579	NDCG: 0.15787
The time for epoch 742 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.11548, F1-score: 0.087182 	 Precision: 0.05695	 Recall: 0.18587	NDCG: 0.15785
The time for epoch 743 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.23661, F1-score: 0.087227 	 Precision: 0.05697	 Recall: 0.18597	NDCG: 0.15792
The time for epoch 744 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.25016, F1-score: 0.087187 	 Precision: 0.05694	 Recall: 0.18595	NDCG: 0.15787
The time for epoch 745 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.52821, F1-score: 0.087210 	 Precision: 0.05695	 Recall: 0.18604	NDCG: 0.15792
The time for epoch 746 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.91880, F1-score: 0.087136 	 Precision: 0.05691	 Recall: 0.18586	NDCG: 0.15785
The time for epoch 747 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.10829, F1-score: 0.087102 	 Precision: 0.05690	 Recall: 0.18564	NDCG: 0.15779
The time for epoch 748 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.71441, F1-score: 0.087163 	 Precision: 0.05693	 Recall: 0.18591	NDCG: 0.15791
The time for epoch 749 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.28754, F1-score: 0.087206 	 Precision: 0.05695	 Recall: 0.18607	NDCG: 0.15792
The time for epoch 750 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.16543, F1-score: 0.087184 	 Precision: 0.05694	 Recall: 0.18590	NDCG: 0.15782
The time for epoch 751 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.17033, F1-score: 0.087094 	 Precision: 0.05689	 Recall: 0.18564	NDCG: 0.15769
The time for epoch 752 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.00930, F1-score: 0.087106 	 Precision: 0.05689	 Recall: 0.18575	NDCG: 0.15788
The time for epoch 753 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.12753, F1-score: 0.087081 	 Precision: 0.05688	 Recall: 0.18567	NDCG: 0.15777
The time for epoch 754 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.96309, F1-score: 0.087028 	 Precision: 0.05683	 Recall: 0.18568	NDCG: 0.15778
The time for epoch 755 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.59768, F1-score: 0.087072 	 Precision: 0.05687	 Recall: 0.18570	NDCG: 0.15767
The time for epoch 756 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 67.18422, F1-score: 0.087083 	 Precision: 0.05687	 Recall: 0.18576	NDCG: 0.15783
The time for epoch 757 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.47238, F1-score: 0.087079 	 Precision: 0.05687	 Recall: 0.18574	NDCG: 0.15781
The time for epoch 758 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.17972, F1-score: 0.087056 	 Precision: 0.05686	 Recall: 0.18566	NDCG: 0.15774
The time for epoch 759 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.45986, F1-score: 0.087034 	 Precision: 0.05685	 Recall: 0.18554	NDCG: 0.15773
The time for epoch 760 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 67.19929, F1-score: 0.087058 	 Precision: 0.05686	 Recall: 0.18567	NDCG: 0.15773
The time for epoch 761 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.02174, F1-score: 0.087022 	 Precision: 0.05684	 Recall: 0.18554	NDCG: 0.15772
The time for epoch 762 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.14690, F1-score: 0.087142 	 Precision: 0.05692	 Recall: 0.18576	NDCG: 0.15780
The time for epoch 763 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.96090, F1-score: 0.087091 	 Precision: 0.05690	 Recall: 0.18552	NDCG: 0.15762
The time for epoch 764 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.75064, F1-score: 0.087087 	 Precision: 0.05689	 Recall: 0.18558	NDCG: 0.15779
The time for epoch 765 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.85252, F1-score: 0.087115 	 Precision: 0.05691	 Recall: 0.18562	NDCG: 0.15773
The time for epoch 766 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 65.01434, F1-score: 0.087054 	 Precision: 0.05686	 Recall: 0.18558	NDCG: 0.15773
The time for epoch 767 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.06320, F1-score: 0.087036 	 Precision: 0.05685	 Recall: 0.18556	NDCG: 0.15779
The time for epoch 768 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.89851, F1-score: 0.087085 	 Precision: 0.05688	 Recall: 0.18565	NDCG: 0.15775
The time for epoch 769 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.52038, F1-score: 0.087061 	 Precision: 0.05687	 Recall: 0.18556	NDCG: 0.15776
The time for epoch 770 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.70620, F1-score: 0.087132 	 Precision: 0.05691	 Recall: 0.18575	NDCG: 0.15781
The time for epoch 771 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.90271, F1-score: 0.087133 	 Precision: 0.05692	 Recall: 0.18569	NDCG: 0.15783
The time for epoch 772 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 64.31197, F1-score: 0.086954 	 Precision: 0.05682	 Recall: 0.18519	NDCG: 0.15771
The time for epoch 773 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 66.64460, F1-score: 0.087084 	 Precision: 0.05689	 Recall: 0.18562	NDCG: 0.15772
The time for epoch 774 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.78159, F1-score: 0.087055 	 Precision: 0.05686	 Recall: 0.18561	NDCG: 0.15771
The time for epoch 775 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 66.52852, F1-score: 0.087043 	 Precision: 0.05687	 Recall: 0.18546	NDCG: 0.15769
The time for epoch 776 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.48172, F1-score: 0.087082 	 Precision: 0.05689	 Recall: 0.18554	NDCG: 0.15769
The time for epoch 777 is: train time = 00: 00: 55, test time = 00: 00: 08
Loss = 65.32026, F1-score: 0.087119 	 Precision: 0.05691	 Recall: 0.18569	NDCG: 0.15771
The time for epoch 778 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.32909, F1-score: 0.087114 	 Precision: 0.05690	 Recall: 0.18575	NDCG: 0.15774
The time for epoch 779 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.78394, F1-score: 0.087176 	 Precision: 0.05694	 Recall: 0.18586	NDCG: 0.15782
The time for epoch 780 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 65.29471, F1-score: 0.087092 	 Precision: 0.05689	 Recall: 0.18568	NDCG: 0.15763
The time for epoch 781 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.18042, F1-score: 0.087098 	 Precision: 0.05689	 Recall: 0.18568	NDCG: 0.15773
The time for epoch 782 is: train time = 00: 00: 53, test time = 00: 00: 08
Loss = 66.63930, F1-score: 0.087133 	 Precision: 0.05690	 Recall: 0.18587	NDCG: 0.15784
The time for epoch 783 is: train time = 00: 00: 54, test time = 00: 00: 08
Loss = 65.76788, F1-score: 0.087108 	 Precision: 0.05690	 Recall: 0.18574	NDCG: 0.15776
The time for epoch 784 is: train time = 00: 00: 52, test time = 00: 00: 08
Loss = 66.17208, F1-score: 0.087126 	 Precision: 0.05691	 Recall: 0.18575	NDCG: 0.15784
##########################################
Early stop is triggered at 784 epochs.
Results:
best epoch = 634, best recall = 0.18636383812726343, best ndcg = 0.15809711198860704
The best model is saved at ./ultragcn_gowalla.pt
Training end!
END
```
