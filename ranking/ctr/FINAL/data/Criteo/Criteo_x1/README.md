# Criteo_x1

+ **Dataset description:**

  The Criteo dataset is a widely-used benchmark dataset for CTR prediction, which contains about one week of click-through data for display advertising. It has 13 numerical feature fields and 26 categorical feature fields. Following the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, we randomly split the data into 7:2:1\* as the training set, validation set, and test set, respectively. 

  The dataset statistics are summarized as follows:

  | Dataset Split  | Total | #Train | #Validation | #Test | 
  | :--------: | :-----: |:-----: | :----------: | :----: | 
  | Criteo_x1 |  45,840,617     | 33,003,326   |  8,250,124     | 4,587,167     |         

+ **Source:** https://www.kaggle.com/c/criteo-display-ad-challenge/data
+ **Download:** https://huggingface.co/datasets/reczoo/Criteo_x1/tree/main
+ **RecZoo Datasets:** https://github.com/reczoo/Datasets

+ **Used by papers:** 
    - Weiyu Cheng, Yanyan Shen, Linpeng Huang. [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768). In AAAI 2020.
    - Kelong Mao, Jieming Zhu, Liangcai Su, Guohao Cai, Yuru Li, Zhenhua Dong. [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902). In AAAI 2023.
    - Jieming Zhu, Qinglin Jia, Guohao Cai, Quanyu Dai, Jingjie Li, Zhenhua Dong, Ruiming Tang, Rui Zhang. [FINAL: Factorized Interaction Layer for CTR Prediction](https://dl.acm.org/doi/10.1145/3539618.3591988). In SIGIR 2023.

+ **Check the md5sum for data integrity:**
    ```bash
    $ md5sum train.csv valid.csv test.csv
    30b89c1c7213013b92df52ec44f52dc5  train.csv
    f73c71fb3c4f66b6ebdfa032646bea72  valid.csv
    2c48b26e84c04a69b948082edae46f8c  test.csv
    ```
