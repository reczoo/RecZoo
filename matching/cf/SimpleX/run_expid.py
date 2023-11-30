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


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import recbox_version
from recbox import datasets
from datetime import datetime
from recbox.utils import load_config, set_logger, print_to_json, print_to_list
from recbox.utils.torch_utils import seed_everything
from recbox.matching.features import FeatureMap, FeatureEncoder
from recbox.matching.pytorch.dataloaders import h5_generator
import src
import gc
import argparse
import logging
import os
from pathlib import Path
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/', help='The config directory.')
    parser.add_argument('--expid', type=str, help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    
    args = vars(parser.parse_args())
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']

    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # build dataset to h5 and load feature_map
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform h5 data
        feature_encoder = FeatureEncoder(**params)
        if not os.path.exists(feature_map_json):
            datasets.build_dataset(feature_encoder, **params)
        params["train_data"] = os.path.join(data_dir, 'train.h5')
        params["valid_data"] = os.path.join(data_dir, 'valid.h5')
        params["test_data"] = os.path.join(data_dir, 'test.h5') if "test_data" in params else None
        params["item_corpus"] = os.path.join(data_dir, 'item_corpus.h5')
    feature_map = FeatureMap(params['dataset_id'], data_dir, params['query_index'], 
                             params['corpus_index'], params['label_col']['name'])
    feature_map.load(feature_map_json)

    # define the model 
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    # fit the model
    train_gen, valid_gen = h5_generator(feature_map, stage='train', **params)
    model.fit(train_gen, valid_generator=valid_gen, **params)
    model.load_weights(model.checkpoint)

    # evaluate the model
    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(train_gen, valid_gen)
    del valid_gen
    gc.collect()
    
    test_result = dict()
    if params.get("test_data"):
        logging.info('******** Test evaluation ********')
        test_gen = h5_generator(feature_map, stage='test', **params)
        test_result = model.evaluate(train_gen, test_gen)
    
    # save results to csv
    with open(Path(args['config']).stem + '.csv', 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))

