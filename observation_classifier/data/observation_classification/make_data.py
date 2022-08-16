'''
This script  do all needed to make the data ready for training 
- read the data
- do preprocessing
- tokenize the text column
- make embeddings for the tokenized data
- encode the labels 
- split the data to train, val and test
'''
import string
from sys import path
from allennlp import torch
import pandas as pd
import numpy as np
import random
import sys
from observation_classifier.models.observation_classification.miscc.config import cfg, cfg_from_file
from observation_classifier.data.observation_classification.embeddings import read_embeddings, load_label_encoder, text_encoder
import os
from sklearn.preprocessing import LabelEncoder

def clean_names(data, col_idxs, col_names, has_date=True):
    '''
    cleaning the name of the columns 
    - rename the name of the columns to be easier to read
    - removing spaces from the name of the columns
    - fill nan values in date column with 0
    - convert date column to date field
    '''
    for i in range(len(col_idxs)):
        data.rename(columns={ data.columns[col_idxs[i]]: col_names[i] }, inplace = True)

    data.columns = [c.replace(' ', '_') for c in data.columns]
    data.columns = [c.replace('_\(FSIII\)',' ') for c in data.columns]
    if has_date:
        #Convert date column to data datatype
        data['date'] = data['date'].fillna(0, inplace = True)
        data['date'] = pd.to_datetime(data['date'])
    return data

def read_data(path , name, type = ''):
    # Reading different kind of data with different codes because we got different formats
    if type == 'observations':
        obs_path = os.path.join(path, name + '.csv')
        data = pd.read_csv(obs_path )

    return data

def basic_cleaning(data):
    # Convert text to lowercase
    # Remove punctutations from the text field
    if data != '\\N':
        data = data.lower()
        for punctuation in string.punctuation:
            data.replace(punctuation, '') 
    return data


def pre_process(data, col , le , cfg):
    '''
    Doing some pre-processing:
        - removing duplicates
        - Sort the data based on the date
        - encoding labels: from their names to the values
        - tokenizing the text column
    # data: whole data  
    # col : text field
    # le : label encoder    
    '''

    #remove duplicated
    data = data.drop_duplicates(subset =['all_text'])

    # encoding labels
    text = data[col]
    le.fit(data['observationscheme'])    
    labels = le.transform(data['observationscheme'])
    if cfg.EMBEDDING_DIR == '':
        text = tokenization(text)
    return text , labels , data , le

def splitting_data(data, labels, val_data, val_labels, cfg):
    '''
    This splitting function works with both 'health conditions' and 'functional conditions'
    Part 1 works for health conditions
    Part 2 works with both versions
    Splitting Part2:
    - using 20% of the last samples of sorted data for test
    - using the rest of the data for validation and train set
        - using 20% for validation
        - the rest for training
    '''
    n_samples =  len(data)
    if cfg.TRAIN.TEST_NUM != 0:
        Num_test = cfg.TRAIN.TEST_NUM
    else:
        Num_test = int(n_samples * cfg.TRAIN.TEST_RATE / 100)
    
    if cfg.TRAIN.USE_TRANSFER:      
        if cfg.TRAIN.TRANSFER_TYPE == 1:
            x_train, y_train = data[0:n_samples - Num_test] , labels[0:n_samples - Num_test]
            x_test , y_test = data[(n_samples - Num_test) :] , labels[(n_samples - Num_test) :]
            x_val , y_val = val_data , val_labels
        elif cfg.TRAIN.TRANSFER_TYPE == 2:
            x_train, y_train = val_data , val_labels
            x_test , y_test = data[(n_samples - Num_test) :] , labels[(n_samples - Num_test) :]
            x_val , y_val = data[0:n_samples - Num_test] , labels[0:n_samples - Num_test]
        elif cfg.TRAIN.TRANSFER_TYPE == 3:
            x_train, y_train = data[0:n_samples - Num_test] , labels[0:n_samples - Num_test]
            x_test , y_test = data[(n_samples - Num_test) :] , labels[(n_samples - Num_test) :]
            x_val , y_val = val_data , val_labels
            x_train = np.append(x_train, x_val , axis=0)
            y_train = np.append(y_train, y_val, axis=0)
    else:
        x, y = data [0:n_samples - Num_test] , labels[0:n_samples - Num_test]
        x_test , y_test = data [(n_samples - Num_test) :] , labels[(n_samples - Num_test) :]
        x_train , y_train = x[0:len(x) - Num_test] , y[0:len(x) - Num_test]
        x_val , y_val = x[len(x) - Num_test :] , y[len(x) - Num_test :]

    return x_train , y_train , x_val, y_val, x_test, y_test

def prepare_data(cfg):
    '''
    - Reading the lable and text embedding if they are prepared before 
    - Otherwise doing preprocessing and make embeddings, and splitting data to train , val and test sets
    '''
    le = LabelEncoder()
    data, labels = read_embeddings(cfg)
    x_train , y_train , x_val, y_val, x_test, y_test = splitting_data(data, labels, '', '', cfg)
    le = load_label_encoder(le, cfg)
    return x_train, y_train, x_val, y_val, x_test, y_test , le
   

    

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

if __name__ == "__main__":

    main_dir="" 
    proj_dir=os.path.join(main_dir,'/models/observation_classification')    
    cfg_file = os.path.join(proj_dir,'cfg/multi_lingual/LaBSE_XGBoost_test.yml')
    data_dir = os.path.join(main_dir, 'Text_Classification/data/Observation/all_data/v4')
    
    cfg_from_file(cfg_file)  

    cfg.DATA_DIR = data_dir
    cfg.EMBEDDING_DATA_NAME = ''
    cfg.LABEL_ENC_NAME = ''   
    cfg.OUTPUT_DIR = '' 
    cfg.VIS_DATA = False   
    cfg.DATASET_NAME = 'ObservationData_Combined_QA_comments'
    cfg.manualSeed = random.randint(1, 10000)
    cfg.EMBEDDING_DIR = ''
    random.seed(cfg.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPU_ID) 
    x_train , y_train , x_val, y_val, x_test, y_test, le = prepare_data(cfg)
