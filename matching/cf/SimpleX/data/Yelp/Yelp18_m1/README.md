# Yelp18_m1

+ **Dataset description:**

  The data statistics are summarized as follows:

  | Dataset ID          | #Users | #Items | #Interactions |  #Train   |  #Test  | Density |
  | :-------: | :----: | :----: | :-----------: | :-------: | :-----: | :-----: |
  | Yelp18_m1 | 31,668 | 38,048 |   1,561,406   | 1,237,259 | 324,147 | 0.00130 |


+ **Data format:**  
user_id item1 item2 ...

+ **Source:** https://www.yelp.com/dataset
+ **Download:** https://huggingface.co/datasets/reczoo/Yelp18_m1/tree/main
+ **RecZoo Datasets:** https://github.com/reczoo/Datasets

+ **Used by papers:** 
    - Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126). In SIGIR 2020.
    - Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, Xiuqiang He. [SimpleX: A Simple and Strong Baseline for Collaborative Filtering](https://arxiv.org/abs/2109.12613). In CIKM 2021.
    - Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, Xiuqiang He. [UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation](https://arxiv.org/abs/2110.15114). In CIKM 2021.

+ **Check the md5sum for data integrity:**
    ```bash
    $ md5sum *.txt
    520fe559761ff2c654629201c807f353  item_list.txt
    0d57d7399862c32152b045ec5d2698e7  test.txt
    1b8b5d22a227e01d6de002c53d32b4c4  train.txt
    ae4f810cd6e827f10fc418753c7d92f9  user_list.txt
    ```
