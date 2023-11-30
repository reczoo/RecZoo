# Frappe_x1

+ **Dataset description:**

  The Frappe dataset contains a context-aware app usage log, which comprises 96203 entries by 957 users for 4082 apps used in various contexts. It has 10 feature fields including user_id, item_id, daytime, weekday, isweekend, homework, cost, weather, country, city. The target value indicates whether the user has used the app under the context. Following the [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) work, we randomly split the data into 7:2:1 as the training set, validation set, and test set, respectively. 

  The dataset statistics are summarized as follows:

  | Dataset Split  | Total | #Train | #Validation | #Test | 
  | :--------: | :-----: |:-----: | :----------: | :----: | 
  | Frappe_x1 |  288,609   | 202,027  |  57,722    | 28,860    |         

+ **Source:** https://www.baltrunas.info/context-aware
+ **Download:** https://huggingface.co/datasets/reczoo/Frappe_x1/tree/main
+ **RecZoo Datasets:** https://github.com/reczoo/Datasets

+ **Used by papers:** 
  - Weiyu Cheng, Yanyan Shen, Linpeng Huang. [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768). In AAAI 2020.
  - Kelong Mao, Jieming Zhu, Liangcai Su, Guohao Cai, Yuru Li, Zhenhua Dong. [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902). In AAAI 2023.
  - Jieming Zhu, Qinglin Jia, Guohao Cai, Quanyu Dai, Jingjie Li, Zhenhua Dong, Ruiming Tang, Rui Zhang. [FINAL: Factorized Interaction Layer for CTR Prediction](https://dl.acm.org/doi/10.1145/3539618.3591988). In SIGIR 2023.

+ **Check the md5sum for data integrity:**
  ```bash
  $ md5sum train.csv valid.csv test.csv
  ba7306e6c4fc19dd2cd84f2f0596d158 train.csv
  88d51bf2173505436d3a8f78f2a59da8 valid.csv
  3470f6d32713dc5f7715f198ca7c612a test.csv
  ```
