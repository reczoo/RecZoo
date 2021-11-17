## UltraGCN_Amazon_Electronics_x0

A notebook to benchmark UltraGCN on Amazon_Electronics dataset.

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
We follow the data split and preprocessing steps in NBPO.We directly transform the formats of the data from their [repo](https://github.com/Wenhui-Yu/NBPO).

### Code
Due to the conciseness of UltraGCN designs, we adopt the single file style to make the code clear and easy to be validated. All codes are in the file "main.py" with a configuration file "dataset_name_config.ini". The reproduction is very easy:

First, set your parameters in the file "dataset_name_config.ini". See "amazon_config.ini" for reference.


```bash
python main.py --config_file electronics_config.ini
```

### Results
Recall@20: 0.16216122741943766

NDCG@20 = 0.10432259412356267


### Logs
```bash


```
