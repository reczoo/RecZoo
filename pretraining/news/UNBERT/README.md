# UNBERT

UNBERT is a BERT-based user-news matching model that leverages the use of the successful BERT pre-training technique for news recommendation. In contrast to existing research, the UNBERT model not only leverages the pre-trained model with rich language knowledge to enhance textual representation, but also captures multi-grained user-news matching signals at both word-level and newslevel.

> Qi Zhang, Jingjie Li, Qinglin Jia, Chuyuan Wang, Jieming Zhu, Zhaowei Wang, Xiuqiang He. [UNBERT: User-News Matching BERT for News Recommendation](https://www.ijcai.org/proceedings/2021/462), in IJCAI 2021.

## Requirements
```bash
pip install -r requirements.txt
```

## Data preparation

For the MIND dataset, please download at https://msnews.github.io

File Name | Description
------------- | -------------
data/bert-base-uncased  | pretrained model from huggingface
data/small/train  | MIND-small train dataset
data/small/dev  | MIND-small dev dataset
data/large/train  | MIND-large train dataset 
data/large/dev  | MIND-large dev dataset
data/large/test  | MIND-large test dataset

## Usage

```python
python run.py --mode train --split small --root ./data/ --pretrain data/bert-base-uncased/
```

For more hyper-parameter settings, please refer to `run.py`.
