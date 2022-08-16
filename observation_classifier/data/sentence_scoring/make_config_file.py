import os
import json


conf_type_choices = [
    'sentence_scoring_experiments'
    'pre_processing',
    'make_sentence_scoring_data',
    'make_subsample_data',
    'sentence_scoring_analysis'
]

conf_type = 'pre_processing'


if conf_type == 'pre_processing':

    #-------------------------------------------------------------
    # Configuration file for preparing data
    # pre_processing.py
    #-------------------------------------------------------------
    args = {
        "output_path": "/data/Observation/all_data/v4/sentence_scoring/new_version/raw_data",
        "data_name": "ObservationData_cleand.csv",
        "converter_name": "convert_schemes_pattern.csv",
        "superuser_data": True,
        "filter_num_tokens": True,
        "token_threshold": 5,
        "converter_path": "/data/npy_embeddings/sub_samples/classes",
        "data_path": "/data/Observation/all_data/v4",
        "save_data_flag": False,
        "superusers_file_name":"superusers.csv",
        "out_dir": "observation_classifier/data" # don't change this. It is constant and for saving the config file
    }
if conf_type == 'make_sentence_scoring_data':
    args = {
        #"data_path": "/Sentence_scoring/interim/not_superuserdata",
        "data_path": "/data/Observation/all_data/v4/sentence_scoring/raw_data",
        "data_file_name": "ObservationData_cleaned_relavant_originalNames_preprocess_superusers_filterLen.csv",
        "comment_file_name": "ObservationComments_v4.csv",
        "save_data_flag": False,
        #"output_path": "/Sentence_scoring/interim/superuserdata",
        "output_path":"/data/Observation/all_data/v4/sentence_scoring/new_version/raw_data",

        # if using const splited test for combining fields
        "make_pos_neg_data":True,
        "const_split_data": "ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted_const_test_trainval_split.pkl",
        "pos_neg_data_name": "ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted.csv",

        # combining fields params 
        "col_name": "ObservationAnswer",
        "combine_version": 1, # choices = [0,1,2] representing comma (0), SEP (1), CLS SEP (2)
        "combine_type":"SchemeAnswer",

        # make const test set
        "const_test_split_data": True,
        "const_num_test": 16000, # size of the constant test set
        
        # making pos_neg samples params
        "token_threshold": 5,        
        "apply_superusers": True,
        "superusers_file_name": "superusers.csv",    
        "su_data_path":"/data/Observation/all_data/v4",
        
        "out_dir": "observation_classifier/data/sentence_scoring" # don't change this. It is constant and for saving the config file    
        
        }
if conf_type == 'make_subsample_data':
    args = {
        # data paths 
        "data_name": "ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted",
        "output_path": "/data/Observation/all_data/v4/sentence_scoring/new_version/raw_data",
        "data_path": "/data/Observation/all_data/v4/sentence_scoring/new_version/raw_data",

        # If we are using constant test set for all kinds of data
        "const_split": True,
        "cont_split_dir": "/Observations/Sentence_scoring/interim/superuserdata/sep_by_SEP/Scheme_Answer_Combination/neg_rate1",
        "const_split_data": "SchemeAnswer_ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted_const_test_trainval_split.pkl",
        "test_rate":0.2, # rate of the test and val sets
        "sub_num_list": [ 0, 10000, 20000, 50000 ], # different sizes for subsampling. 0 for all_data
        "out_dir": "observation_classifier/data/sentence_scoring", # don't change this. It is constant and for saving the config file
        "make_dataset_format":True, # converting data to DataSet format python
        "combine_type":"SchemeAnswer",
        "save_data_flag": True
        }
if conf_type == 'sentence_scoring_experiments':
    #-------------------------------------------------------------
    # Configuration file for sentence scoring experiments
    # (sent_score_exp.py) and DataTransformer.py
    #-------------------------------------------------------------
    args = {
        "checkpoint" : "Maltehb/aelaectra-danish-electra-small-cased", 
        "data_dir" : "/data/Observation/all_data/v4/sentence_scoring/new_version", #'/Sentence_scoring/interim/superuserdata',
        "separator" : "sep_by_SEP", #'sep_by_SEP/Scheme_Answer_Combination/neg_rate1',
        "tokenized_dir" : "aelaectra-danish-electra-small-cased", 
        "file_name" : 'ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted_const_test_trainval_split.pkl',
        "tokenize_data" : True,
        "out_dir": "observation_classifier/models/sentence_scoring", # don't change this. It is constant and for saving the config file

        # Training arguments
        "output_dir":"/output/models/Sentence_scoring/new_version/SchemeAnswer/aelaectra-danish-electra-small-cased/all_data",
        "learning_rate":5e-3,
        "per_device_train_batch_size":256,
        "per_device_eval_batch_size":256,
        "num_train_epochs":500,
        "weight_decay":0.00001,
        "save_metrics": False,
        "freeze_layers": True,
        "last_no_layers":4,
        "num_warmup_steps": 1000,
        "num_cycles": 5,
        "lr_schedule": None, # "cosine_with_restart",

        # DataTransformer
        "save_datadict" : True,
        "tokenize_name": "SchemeAnswer_All",
        "logging_dir": "logging",
        "evaluation_steps": 256,
        "logging_steps":256,
        "evaluation_strategy": "steps" #"epoch"
        
    }
if conf_type == 'sentence_scoring_analysis':
    args = {
        "num_labels": 2,
        "k":3,
        "eval_bs": 256,
        "checkpoint" : "/output/models/Sentence_scoring/new_version/SchemeAnswer/mMiniLMv2-L6-H384-distilled-from-XLMR-Large/all_data/Epoch_20_lr_0.002_26_04_2022_10:38",
        "log_dir": "26.04.2022_08:52:08",
        "log_name" : "events.out.tfevents.1650955940.EP-Workstation.236430.0",
        "tok_checkpoint": "Maltehb/aelaectra-danish-electra-small-cased",
        "data_dir" : '/data/Observation/all_data/v4/sentence_scoring/new_version/raw_data/seed100/neg_rate1',
        "data_name" : 'ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted_const_test_trainval_split.pkl',
        "out_dir": "observation_classifier/data/sentence_scoring", # don't change this. It is constant and for saving the config file
        "col_name": "ObservationAnswer",
        "combine_version": 1,
        "combine_type": "SchemeAnswer",
        "eval_list": ['precision_at_k','eval_metrics', 'training_analysis'],
        "conf_list" : [0.4,0.5, 0.6],
        "save_figs": True,
    }


if __name__=="__main__":
    file_name = f'{conf_type}_config.json'
    file_path = os.path.join(args['out_dir'], file_name)
    json_config = json.dumps(args)
    with open(file_path, 'w') as jsonfile:
        json.dump(args, jsonfile, indent=2)
 