# MovielensLatest_x1

+ **Dataset description:**
  
  The MovieLens dataset consists of users' tagging records on movies. The task is formulated as personalized tag recommendation with each tagging record (user_id, item_id, tag_id) as an data instance. The target value denotes whether the user has assigned a particular tag to the movie. Following the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, we randomly split the data into 7:2:1 as the training set, validation set, and test set, respectively. 

  The dataset statistics are summarized as follows:

  | Dataset Split  | Total | #Train | #Validation | #Test | 
  | :--------: | :-----: |:-----: | :----------: | :----: | 
  | MovielensLatest_x1 |  2,006,859   | 1,404,801  |  401,373    | 200,686   |  

+ **Source:** https://grouplens.org/datasets/movielens
+ **Download:** https://huggingface.co/datasets/reczoo/MovielensLatest_x1/tree/main
+ **RecZoo Datasets:** https://github.com/reczoo/Datasets

+ **Used by papers:**
  - Weiyu Cheng, Yanyan Shen, Linpeng Huang. [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768). In AAAI 2020.
  - Kelong Mao, Jieming Zhu, Liangcai Su, Guohao Cai, Yuru Li, Zhenhua Dong. [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902). In AAAI 2023.
  - Jieming Zhu, Qinglin Jia, Guohao Cai, Quanyu Dai, Jingjie Li, Zhenhua Dong, Ruiming Tang, Rui Zhang. [FINAL: Factorized Interaction Layer for CTR Prediction](https://dl.acm.org/doi/10.1145/3539618.3591988). In SIGIR 2023.
  
+ **Check the md5sum for data integrity:**
  ```bash
  $ md5sum train.csv valid.csv test.csv
  efc8bceeaa0e895d566470fc99f3f271 train.csv
  e1930223a5026e910ed5a48687de8af1 valid.csv
  54e8c6baff2e059fe067fb9b69e692d0 test.csv
  ```
