## UltraGCN_amazonbooks_x0

A notebook to benchmark UltraGCN on Yelp2018 dataset.

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
python main.py --config_file yelp2018_config.ini
```

### Results
Recall@20: 0.06810916655039237

NDCG@20 = 0.05581510919406059

### Logs
```bash
Loss = 73.96327, F1-score: 0.041150 	 Precision: 0.02985	 Recall: 0.06622	NDCG: 0.05449
The time for epoch 73 is: train time = 00: 01: 04, test time = 00: 00: 08
Loss = 73.54290, F1-score: 0.041182 	 Precision: 0.02986	 Recall: 0.06635	NDCG: 0.05470
The time for epoch 74 is: train time = 00: 01: 04, test time = 00: 00: 09
Loss = 75.74036, F1-score: 0.041173 	 Precision: 0.02989	 Recall: 0.06614	NDCG: 0.05460
The time for epoch 75 is: train time = 00: 01: 04, test time = 00: 00: 09
Loss = 75.08773, F1-score: 0.041182 	 Precision: 0.02986	 Recall: 0.06634	NDCG: 0.05467
##########################################
Early stop is triggered at 75 epochs.
Results:
best epoch = 60, best recall = 0.06810916655039237, best ndcg = 0.05581510919406059
The best model is saved at ./ultragcn_yelp2018.pt
Training end!
END

```