'''
Gets all arguments from cfg amd misc folders for different kinds of experiments
to be able to run the experiments, you just need to get the right config file 
and change cfg_file directory.
There is a config file for each kind of the experiments in .sh format and in the experiment folder
The result of each experiment is saved in experiment folder and in .out format
'''
from observation_classifier.models.observation_classification.miscc.config import cfg, cfg_from_file
from train import train , validate , save_model
import os
import sys
import random
import datetime
import dateutil.tz
import argparse
import torch
import numpy as np

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='laser_XGBoost.yml', type=str)
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', help='directory contains csv data', type=str, default='')
    parser.add_argument('--embd_data_name', dest='embd_data_name', type=str, help='augmented data name',default='')
    parser.add_argument('--train_model', dest='train_model', type=str, help='trained model',default='')
    parser.add_argument('--output_dir', dest='output_dir', help='Output directory to save results', type=str, default='')
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='observations',default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--embd_dir' , dest='embd_dir' , help='The directory to the embedding files', type=str, default='')
    parser.add_argument('--label_enc_name' , dest='label_enc_name' , help='The dir to the label_encoder', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.VIS_DATA = args.vis_data_flag
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id        
    else:
        cfg.CUDA = False
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.embd_data_name != '':
        cfg.EMBEDDING_DATA_NAME = args.embd_data_name
    if args.train_model != '':
        cfg.TRAIN.MODEL = args.train_model
    if args.label_enc_name != '':
        cfg.LABEL_ENC_NAME = args.label_enc_name
    if args.use_transfer :
        cfg.TRAIN.TRANSFER_TYPE = args.transfer_type
    if args.output_dir != '':
        cfg.OUTPUT_DIR = args.output_dir
    if args.dataset_name != '':
        cfg.DATASET_NAME = args.dataset_name
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.embd_dir != '' :
        cfg.EMBEDDING_DIR = args.embd_dir
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPU_ID)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    if cfg.TRAIN.FLAG:
        model = train(cfg)   
        save_model(model, cfg)
    
    else:
        if cfg.TRAIN.MODEL =='':
            raise ValueError('Error: The path for model not found!')
        else:
            validate(cfg)