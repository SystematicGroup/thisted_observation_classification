# Split embeddings into train/val/test set
import os
import pandas as pd
import numpy as np
from observation_classifier.data.pre_processing import read_json_file, save_data

def make_const_split_embeddings(args,type, save=False):
    """Generate fixed embedding-sets for New Observation-task

    Args:
        args (dict): Config including argument "emb_data_dir" for loading the full embeddings, and saving the splitted embeddings with labels.
        type (string): The type of embedding used(Must be the name of an embedded file in 'data_dir')
        save (bool, optional): Save test set. Defaults to False.

    Output:
        trian_eval (pd.DataFrame): Train and evaluation set (With labels)
        test (pd.DataFrame): Test set (With labels)
    """

    data_dir = args['emb_data_dir']

    if type == 'BERT':
        labels = pd.DataFrame(np.load(f'{data_dir}/labels_ObservationData_Combined_QA_comments_superusers_lenmorethan5_{type}_embeddings.npy'))
        input_ids = pd.DataFrame(np.load(f'{data_dir}/ObservationData_Combined_QA_comments_superusers_lenmorethan5_{type}_embeddings_input_ids.npy'))
        attention_masks = pd.DataFrame(np.load(f'{data_dir}/ObservationData_Combined_QA_comments_superusers_lenmorethan5_{type}_embeddings_attention_masks.npy'))
        col = []
        for inp in input_ids.columns:
            col.append(f'{inp}_id')

        input_ids.columns = col

        data_label = input_ids.assign(labels = labels[0])
        for i,val in enumerate(attention_masks):
           data_label[f'{i}_att'] = attention_masks[i]

        test_size = int(np.round(len(input_ids)*args['TEST_PERC']))
        train_eval = data_label.head(len(data_label)-test_size)
        test = data_label.tail(test_size)

    else:

        labels = pd.DataFrame(np.load(f'{data_dir}/labels_ObservationData_Combined_QA_comments_superusers_lenmorethan5_{type}_embeddings.npy'))
        data = pd.DataFrame(np.load(f'{data_dir}/ObservationData_Combined_QA_comments_superusers_lenmorethan5_{type}_embeddings.npy'))
        
        test_size = int(np.round(len(data)*args['TEST_PERC']))
        data_label = data.assign(labels = labels[0])
        train_eval = data_label.head(len(data_label)-test_size)
        test = data_label.tail(test_size)

    if save:
        save_data(train_eval, args['emb_data_dir'], name = f'newobs_{type}_train_eval.csv')
        save_data(test, args['emb_data_dir'], name = f'newobs_{type}_test_fixed.csv')

    return train_eval, test

if __name__=="__main__":
    args = read_json_file(f"/observation_classifier/data/observation_classification/new_observations/newobs_config",os.getcwd())
    _ = make_const_split_embeddings(args, 'tfidf', save=True) #'tfidf', 'LaBSE', 'roberta','BERT'
