from __future__ import division
from __future__ import print_function
from pickle import FALSE

'''
This file contains the default values for experiments of text classification
'''
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: functional_conditions, health_conditions, observations
__C.DATASET_NAME = 'functional_conditions'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.OUTPUT_DIR = ''
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 4
__C.STD_VALIDATION = False # using standard data for validation
__C.VIS_DATA = True # loading data from visitors
__C.AUG_DATA_NAME = ''
__C.EMBEDDING_DATA_NAME = ''
__C.EMBEDDING_DIR = ''
__C.EMBEDDING_AUG = False
__C.LABEL_ENC_NAME = ''



# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.MAX_EPOCH = 10000
__C.TRAIN.LR = 0.1
__C.TRAIN.GAMMA = 0.0
__C.TRAIN.EARLY_STOPPING = 10
__C.TRAIN.CLASSIFIER = 'XGBOOST'
__C.TRAIN.TEST_RATE = 0
__C.TRAIN.MODEL = ''
__C.TRAIN.USE_TRANSFER = False
__C.TRAIN.TRANSFER_TYPE = 1
__C.TRAIN.MODEL = '' # load saved trained model
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.TEST_NUM = 0




__C.TEXT = edict()
__C.TEXT.EMBEDDING_MODEL = ''   # 'LASER', 'LaSBE' , ...
__C.TEXT.MULTILINGUAL_TYPE = ''



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b: # b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f , Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)