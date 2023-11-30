# =========================================================================
# Copyright (C) 2020-2023. The SimpleX Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


from torch import nn
import torch
import torch.nn.functional as F
from recbox.matching.pytorch.models import MatchingModel
from recbox.matching.pytorch.layers import EmbeddingLayer


class MF(MatchingModel):
    def __init__(self, 
                 feature_map, 
                 model_id="MF",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 num_negs=1,
                 regularizer=None,
                 embedding_dropout=0,
                 similarity_score="dot",
                 **kwargs):
        super(MF, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer,
                                 num_negs=num_negs,
                                 embedding_initializer=embedding_initializer,
                                 **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        assert similarity_score in ["dot", "cosine", "sigmoid"]
        self.dropout = nn.Dropout(embedding_dropout)
        self.compile(lr=learning_rate, **kwargs)
        
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        user_vecs = self.user_tower(user_dict)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1), 
                           user_vecs.unsqueeze(-1)).squeeze(-1)
        if self.similarity_score == "sigmoid":
            y_pred = y_pred.sigmoid()
        loss = self.get_total_loss(y_pred, labels)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_vec = self.embedding_layer(user_inputs, feature_source="user")
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec = self.embedding_layer(item_inputs, feature_source="item")
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        return item_vec

