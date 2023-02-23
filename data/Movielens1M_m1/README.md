The preprocessed dataset is provided by the paper "Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters" in ICML'20.

Data format:  
The original json data has the following format: [[item1, item2], [item3, item4, item5], ...]  
Each user corresponds to a list of interacted items. The user_id can be infered from its index.  

We convert the data to the txt format as follows:  
user_id item1 item2 ...

For more information about the dataset, please visit https://openbenchmark.github.io/BARS/datasets/README.html

