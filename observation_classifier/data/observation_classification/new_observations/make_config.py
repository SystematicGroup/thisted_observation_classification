import os
import json

args = {
    'TEST_PERC': 0.2,
    'obs_data_dir':'/data/Observation/all_data/v4',
    'new_obs_data_dir': '/data/Observation/all_data/v4/new_obs/data',
    'new_obs_trainval_data' : 'newobs_trainval_unfiltered_sub_100000.csv',
    'new_obs_test_data':'newobs_testset_fixed_sub_100000.csv',
    'new_obs_original_classes_dir':'/data/Observation/all_data/v4/sentence_scoring/new_obs',
    'new_obs_original_classes':'newobs_original_classes.npy',
    'embedding_data_dir': '/interim/Normal_Combination/all_samples/newobs/embeddings',
    'embed_type':'LaBSE',
    "proj_dir": "/models/observation_classification",
    "col_name" :'observationscheme',
    # Training model
    "model_type":"huggingface",  #"sklearn", 
    "last_no_layers":8,
    
    "sklearn_model_name":"sklearn.linear_model.LogisticRegression", # if model_type is sklearn
    "checkpoint_dir": "/results" ,
    "checkpoint": 'Maltehb/aelaectra-danish-electra-small-cased', #'Maltehb/danish-bert-botxo', # if model_type is huggingface
    "excluded_checkpoint" :"excluded_finetuned_model",
    "exclude_model":False, #set to True if you have a finetuned model on the data without the new obsarvation
    "excluded_training_param_dict":{
        "freeze_layers": True,
        "last_no_layers": 8,
        "num_epochs":100,
        "lr":0.005,
        "weight_decay":0.01,
        "bs":256
    },

    "included_training_param_dict":{
        "freeze_layers": True,
        "last_no_layers": 8,
        "num_epochs":200,
        "lr":0.005,
        "weight_decay":0.01,
        "bs":256
    },

    "param_dict": {
        "C": 0.01,
        "class_weight": "balanced",
        "max_iter": 10000,
        "penalty": "l2",
        "random_state" : 42
    },
    
    "kfold":5,
    "obsscheme_exclude": [33],
    "obsscheme_exclude_name": ["medicinscreening"], 
    "max_excluded_samples":71,
    "exclude_samples_interval":20,
    "hf_training_version":2,
    "added_num": 20
}

if __name__=="__main__":
    file_name = f'newobs_config.json'
    file_path = os.path.join("observation_classifier/data/observation_classification/new_observations", file_name)
    json_config = json.dumps(args)
    with open(file_path, 'w') as jsonfile:
        json.dump(args, jsonfile, indent=2)