# Avazu_x1

+ **Dataset description:**

  This dataset contains about 10 days of labeled click-through data on mobile advertisements. It has 22 feature fields including user features and advertisement attributes. As with the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, the data are randomly split into 7:1:2 as the training set, validation set, and test set, respectively. 

  The dataset statistics are summarized as follows:

  | Dataset  | Total | #Train | #Validation | #Test | 
  | :--------: | :-----: |:-----: | :----------: | :----: | 
  | Avazu_x1 |  40,428,967     | 28,300,276   |  4,042,897     |  8,085,794    |     

+ **Source:** https://www.kaggle.com/c/avazu-ctr-prediction/data
+ **Download:** https://huggingface.co/datasets/reczoo/Avazu_x1/tree/main
+ **RecZoo Datasets:** https://github.com/reczoo/Datasets

+ **Used by papers:** 
    - Weiyu Cheng, Yanyan Shen, Linpeng Huang. [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768). In AAAI 2020.
    - Kelong Mao, Jieming Zhu, Liangcai Su, Guohao Cai, Yuru Li, Zhenhua Dong. [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902). In AAAI 2023.
    - Jieming Zhu, Qinglin Jia, Guohao Cai, Quanyu Dai, Jingjie Li, Zhenhua Dong, Ruiming Tang, Rui Zhang. [FINAL: Factorized Interaction Layer for CTR Prediction](https://dl.acm.org/doi/10.1145/3539618.3591988). In SIGIR 2023.

+ **Check the md5sum for data integrity:**
    ```bash
    $ md5sum train.csv valid.csv test.csv
    f1114a07aea9e996842c71648e0f6395  train.csv
    d9568f246357d156c4b8030fadb8b623  valid.csv
    9e2fe9c48705c9315ae7a0953eb57acf  test.csv
    ```
