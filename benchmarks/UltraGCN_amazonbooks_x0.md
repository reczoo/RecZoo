## UltraGCN_amazonbooks_x0

A notebook to benchmark UltraGCN on Amazonbooks dataset.

Author: Kelong Mao, Tsinghua University

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
python main.py --config_file amazon_config.ini
```

### Results
Recall@20: 0.06809152719270753

NDCG@20 = 0.05558807004339008

### Logs
```bash
###################### UltraGCN ######################
1. Loading Configuration...
load path = ./amazon_ii_constraint_mat object
load path = ./amazon_ii_neighbor_mat object
Load Configuration OK, show them below
Configuration:
{'embedding_dim': 64, 'ii_neighbor_num': 10, 'model_save_path': './ultragcn_amazon.pt', 'max_epoch': 2000, 'enable_tensorboard': True, 'dataset': 'amazon', 'gpu': '3', 'device': device(type='cuda', index=3), 'lr': 0.001, 'batch_size': 1024, 'early_stop_epoch': 15, 'w1': 1e-8, 'w2': 1.0, 'w3': 1.0, 'w4': 1e-8, 'negative_num': 500, 'negative_weight': 500.0, 'gamma': 0.0001, 'lambda': 2.75, 'sampling_sift_pos': False, 'test_batch_size': 2048, 'topk': 20, 'user_num': 52643, 'item_num': 91599}
Total training batches = 2325
2021-01-24 10:57:56.075444: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
The time for epoch 0 is: train time = 00: 00: 45, test time = 00: 00: 29
Loss = 1964.60083, F1-score: 0.002423    Precision: 0.00180      Recall: 0.00370        NDCG: 0.00308
The time for epoch 5 is: train time = 00: 00: 49, test time = 00: 00: 28
Loss = 996.04657, F1-score: 0.002449     Precision: 0.00182      Recall: 0.00373        NDCG: 0.00310
The time for epoch 10 is: train time = 00: 00: 49, test time = 00: 00: 29
Loss = 798.89764, F1-score: 0.017066     Precision: 0.01243      Recall: 0.02720        NDCG: 0.02269
The time for epoch 15 is: train time = 00: 00: 48, test time = 00: 00: 28
Loss = 725.17834, F1-score: 0.028051     Precision: 0.02015      Recall: 0.04614        NDCG: 0.03775
The time for epoch 20 is: train time = 00: 00: 50, test time = 00: 00: 29
Loss = 688.42957, F1-score: 0.033414     Precision: 0.02379      Recall: 0.05610        NDCG: 0.04565
The time for epoch 25 is: train time = 00: 00: 52, test time = 00: 00: 29
Loss = 677.29108, F1-score: 0.036335     Precision: 0.02582      Recall: 0.06130        NDCG: 0.04988
The time for epoch 30 is: train time = 00: 00: 55, test time = 00: 00: 29
Loss = 665.36554, F1-score: 0.037670     Precision: 0.02673      Recall: 0.06380        NDCG: 0.05215
The time for epoch 35 is: train time = 00: 00: 53, test time = 00: 00: 28
Loss = 664.83655, F1-score: 0.038583     Precision: 0.02732      Recall: 0.06563        NDCG: 0.05374
The time for epoch 40 is: train time = 00: 00: 50, test time = 00: 00: 29
Loss = 675.28125, F1-score: 0.038760     Precision: 0.02743      Recall: 0.06606        NDCG: 0.05400
The time for epoch 45 is: train time = 00: 00: 51, test time = 00: 00: 30
Loss = 660.45544, F1-score: 0.039383     Precision: 0.02789      Recall: 0.06700        NDCG: 0.05479
The time for epoch 50 is: train time = 00: 00: 49, test time = 00: 00: 29
Loss = 656.82434, F1-score: 0.039214     Precision: 0.02774      Recall: 0.06686        NDCG: 0.05447
The time for epoch 51 is: train time = 00: 00: 45, test time = 00: 00: 28
Loss = 653.54370, F1-score: 0.039201     Precision: 0.02772      Recall: 0.06694        NDCG: 0.05469
The time for epoch 52 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 667.80670, F1-score: 0.039392     Precision: 0.02787      Recall: 0.06714        NDCG: 0.05489
The time for epoch 53 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 640.53198, F1-score: 0.039488     Precision: 0.02794      Recall: 0.06729        NDCG: 0.05496
The time for epoch 54 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 651.38806, F1-score: 0.039581     Precision: 0.02801      Recall: 0.06744        NDCG: 0.05529
The time for epoch 55 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 666.87726, F1-score: 0.039567     Precision: 0.02797      Recall: 0.06757        NDCG: 0.05511
The time for epoch 56 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 648.36584, F1-score: 0.039376     Precision: 0.02785      Recall: 0.06716        NDCG: 0.05494
The time for epoch 57 is: train time = 00: 00: 49, test time = 00: 00: 28
Loss = 653.93457, F1-score: 0.039622     Precision: 0.02802      Recall: 0.06763        NDCG: 0.05532
The time for epoch 58 is: train time = 00: 00: 49, test time = 00: 00: 29
Loss = 655.66028, F1-score: 0.039304     Precision: 0.02781      Recall: 0.06697        NDCG: 0.05463
The time for epoch 59 is: train time = 00: 00: 49, test time = 00: 00: 30
Loss = 649.84888, F1-score: 0.039257     Precision: 0.02775      Recall: 0.06704        NDCG: 0.05468
The time for epoch 60 is: train time = 00: 00: 50, test time = 00: 00: 31
Loss = 668.93256, F1-score: 0.039653     Precision: 0.02807      Recall: 0.06752        NDCG: 0.05509
The time for epoch 61 is: train time = 00: 00: 50, test time = 00: 00: 30
Loss = 662.43909, F1-score: 0.039352     Precision: 0.02784      Recall: 0.06708        NDCG: 0.05482
The time for epoch 62 is: train time = 00: 00: 45, test time = 00: 00: 28
Loss = 652.39587, F1-score: 0.039576     Precision: 0.02799      Recall: 0.06754        NDCG: 0.05520
The time for epoch 63 is: train time = 00: 00: 49, test time = 00: 00: 29
Loss = 655.24707, F1-score: 0.039328     Precision: 0.02779      Recall: 0.06724        NDCG: 0.05469
The time for epoch 64 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 672.68121, F1-score: 0.039276     Precision: 0.02778      Recall: 0.06702        NDCG: 0.05478
The time for epoch 65 is: train time = 00: 00: 51, test time = 00: 00: 29
Loss = 654.69763, F1-score: 0.039534     Precision: 0.02798      Recall: 0.06735        NDCG: 0.05511
The time for epoch 66 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 665.98511, F1-score: 0.039464     Precision: 0.02792      Recall: 0.06726        NDCG: 0.05492
The time for epoch 67 is: train time = 00: 00: 51, test time = 00: 00: 29
Loss = 653.01587, F1-score: 0.039737     Precision: 0.02811      Recall: 0.06778        NDCG: 0.05530
The time for epoch 68 is: train time = 00: 00: 51, test time = 00: 00: 28
Loss = 650.18433, F1-score: 0.039571     Precision: 0.02799      Recall: 0.06751        NDCG: 0.05498
The time for epoch 69 is: train time = 00: 00: 51, test time = 00: 00: 29
Loss = 655.08264, F1-score: 0.039582     Precision: 0.02799      Recall: 0.06757        NDCG: 0.05509
The time for epoch 70 is: train time = 00: 00: 50, test time = 00: 00: 29
Loss = 655.69720, F1-score: 0.039447     Precision: 0.02790      Recall: 0.06730        NDCG: 0.05465
The time for epoch 71 is: train time = 00: 00: 50, test time = 00: 00: 30
Loss = 658.87341, F1-score: 0.039769     Precision: 0.02814      Recall: 0.06809        NDCG: 0.05556
The time for epoch 72 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 640.42426, F1-score: 0.039542     Precision: 0.02797      Recall: 0.06746        NDCG: 0.05497
The time for epoch 73 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 646.27881, F1-score: 0.039747     Precision: 0.02812      Recall: 0.06776        NDCG: 0.05542
The time for epoch 74 is: train time = 00: 00: 48, test time = 00: 00: 28
Loss = 648.02551, F1-score: 0.039410     Precision: 0.02787      Recall: 0.06729        NDCG: 0.05495
The time for epoch 75 is: train time = 00: 00: 50, test time = 00: 00: 28
Loss = 654.54309, F1-score: 0.039647     Precision: 0.02804      Recall: 0.06763        NDCG: 0.05506
The time for epoch 76 is: train time = 00: 00: 51, test time = 00: 00: 28
Loss = 653.28430, F1-score: 0.039413     Precision: 0.02788      Recall: 0.06722        NDCG: 0.05492
The time for epoch 77 is: train time = 00: 00: 50, test time = 00: 00: 29
Loss = 651.10693, F1-score: 0.039277     Precision: 0.02779      Recall: 0.06698        NDCG: 0.05484
The time for epoch 78 is: train time = 00: 00: 48, test time = 00: 00: 28
Loss = 653.71399, F1-score: 0.039479     Precision: 0.02792      Recall: 0.06739        NDCG: 0.05510
The time for epoch 79 is: train time = 00: 00: 50, test time = 00: 00: 29
Loss = 656.40186, F1-score: 0.039773     Precision: 0.02816      Recall: 0.06768        NDCG: 0.05540
The time for epoch 80 is: train time = 00: 00: 48, test time = 00: 00: 31
Loss = 646.55304, F1-score: 0.039695     Precision: 0.02809      Recall: 0.06767        NDCG: 0.05523
The time for epoch 81 is: train time = 00: 00: 50, test time = 00: 00: 29
Loss = 658.48914, F1-score: 0.039447     Precision: 0.02789      Recall: 0.06734        NDCG: 0.05477
The time for epoch 82 is: train time = 00: 00: 51, test time = 00: 00: 29
Loss = 665.40594, F1-score: 0.039743     Precision: 0.02813      Recall: 0.06769        NDCG: 0.05535
The time for epoch 83 is: train time = 00: 00: 51, test time = 00: 00: 36
Loss = 646.29871, F1-score: 0.039600     Precision: 0.02804      Recall: 0.06740        NDCG: 0.05487
The time for epoch 84 is: train time = 00: 00: 51, test time = 00: 00: 33
Loss = 649.15826, F1-score: 0.039408     Precision: 0.02787      Recall: 0.06727        NDCG: 0.05488
The time for epoch 85 is: train time = 00: 00: 50, test time = 00: 00: 31
Loss = 664.64856, F1-score: 0.039584     Precision: 0.02799      Recall: 0.06755        NDCG: 0.05536
The time for epoch 86 is: train time = 00: 00: 51, test time = 00: 00: 29
Loss = 661.28815, F1-score: 0.039747     Precision: 0.02811      Recall: 0.06784        NDCG: 0.05546
##########################################
Early stop is triggered at 86 epochs.
Results:
best epoch = 71, best recall = 0.06809152719270753, best ndcg = 0.05558807004339008
The best model is saved at ./ultragcn_amazon.pt
Training end!
END

```