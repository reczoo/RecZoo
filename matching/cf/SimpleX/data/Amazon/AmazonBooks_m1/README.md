# AmazonBooks_m1

+ **Dataset description:**

  The data statistics are summarized as follows:

  | Dataset ID     | #Users | #Items | #Interactions |   #Train  |  #Test  | Density |
  |:--------------:|:------:|:------:|:-------------:|:---------:|:-------:|:-------:|
  | AmazonBooks_m1 | 52,643 | 91,599 |   2,984,108   | 2,380,730 | 603,378 | 0.00062 |


+ **Data format:**  
user_id item1 item2 ...

+ **Source:** https://cseweb.ucsd.edu/~jmcauley/datasets.html
+ **Download:** https://huggingface.co/datasets/reczoo/AmazonBooks_m1/tree/main
+ **RecZoo Datasets:** https://github.com/reczoo/Datasets

+ **Used by papers:** 
    - Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126). In SIGIR 2020.
    - Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, Xiuqiang He. [SimpleX: A Simple and Strong Baseline for Collaborative Filtering](https://arxiv.org/abs/2109.12613). In CIKM 2021.
    - Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, Xiuqiang He. [UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation](https://arxiv.org/abs/2110.15114). In CIKM 2021.

+ **Check the md5sum for data integrity:**
    ```bash
    $ md5sum *.txt
    5b1125ef3bf4118a7988f1fd8ce52ef9  item_list.txt
    30f8ccfea18d25007ba9fb9aba4e174d  test.txt
    c916ecac04ca72300a016228258b41ed  train.txt
    132f8a5d6d35d5fdde1e0396488be235  user_list.txt
    ```
