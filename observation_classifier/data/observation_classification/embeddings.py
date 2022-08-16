# Making embeddings for text column and encoding data labels
from copyreg import pickle
import pandas as pd
import os
import spacy
from stop_words import get_stop_words
import torch
import sys
import numpy as np 
import random
from observation_classifier.models.observation_classification.miscc.config import cfg, cfg_from_file
from observation_classifier.data.observation_classification.make_data import *
from sklearn.preprocessing import LabelEncoder
from laserembeddings import Laser
from sklearn.feature_extraction.text import TfidfVectorizer
from danish_bert_embeddings import DanishBertEmbeddings
from sentence_transformers import SentenceTransformer

def text_encoder(data, data_transfer, cfg): 
    '''
    This function make embedding for the text column of the data 
    It can be different embeddings and the function gets the type of the embedding from cfg.TEXT.EMBEDDING_MODEL
    If the embedding is made before the function load the embedding file
        - it gets the dir of the embedding file from cfg.EMBEDDING_DIR and cfg.EMBEDDING_DATA_NAME
    Input:
        data: the text data in fomat of dataframe
        data_transfer: for health_conditions
        cfg: config file contains parameters
    Output: 
        embedded text column of the data
    '''
    len_tr = len(data_transfer)
    if cfg.EMBEDDING_DIR != '':
        name = cfg.EMBEDDING_DATA_NAME
        embd_path = os.path.join(cfg.EMBEDDING_DIR , name+'.npy')
        x_features = np.load(embd_path)
    else:        
        if cfg.TEXT.EMBEDDING_MODEL == 'laser':
            embedder = Laser()
            x_features = embedder.embed_sentences(data, lang='da')            
            #np.save('all_embeddings.npy', x_features)
            if len_tr != 0:
                x_features_transfer = embedder.embed_sentences(data_transfer, lang='da')
                return x_features , x_features_transfer            
        elif cfg.TEXT.EMBEDDING_MODEL == 'tfidf':
            embedder = TfidfVectorizer(max_features=1024)
            data = [(' '.join([str(elem) for elem in s])) for s in data]
            embedder = embedder.fit(data)
            x_features = embedder.transform(data)
            x_features = x_features.toarray() 
            if len_tr != 0:
                data_transfer = [(' '.join([str(elem) for elem in s])) for s in data_transfer]
                x_features_transfer = embedder.transform(data_transfer)
                x_features_transfer = x_features_transfer.toarray()
                return x_features , x_features_transfer  
            
        elif cfg.TEXT.EMBEDDING_MODEL == 'bert':
            embedder = DanishBertEmbeddings()
            x_features = data.apply (lambda x: embedder.embed([(' '.join([str(elem) for elem in x]))]))
            x_features = np.vstack(x_features)
            if len_tr != 0:
                x_features_transfer = data_transfer.apply (lambda x: embedder.embed([(' '.join([str(elem) for elem in x]))]))
                x_features_transfer = np.vstack(x_features_transfer)
                return x_features , x_features_transfer
            
        elif cfg.TEXT.EMBEDDING_MODEL == 'multilingual':
            embedder = SentenceTransformer(cfg.TEXT.MULTILINGUAL_TYPE)
            x_features = data.apply (lambda x: embedder.encode([(' '.join([str(elem) for elem in x]))]))
            x_features = np.vstack(x_features)
            if len_tr != 0:
                x_features_transfer = data_transfer.apply (lambda x: embedder.encode([(' '.join([str(elem) for elem in x]))]))
                x_features_transfer = np.vstack(x_features_transfer)
                return x_features , x_features_transfer     
    
    return x_features

def tokenization(text, n=0, dir = ''): 
    '''
    tokenizing with spacy
    converting text to lowercase
    removing punctuations 
    removing danish stop-words
    Input:
        data: the text column of the data
        n: use for subsamples
        dir: the dir for saving the results
    Output:
        tokenized text
    '''
    
    if os.path.exists(dir):
        text = pd.read_pickle(dir)
    else:                
        if torch.cuda.is_available():
            spacy.prefer_gpu()
        nlp = spacy.load("da_core_news_sm")
        da_stopwords = get_stop_words('da')
        text = text.apply(nlp)
        text = text.apply(lambda x: [item.text.lower() for item in x if (item.text not in da_stopwords and item.text.isalpha()== True)])
        # Uncomment to save the tokenized data
        #name = 'random_subsample_%d_tokenized.pkl' %(n)
        #text.to_pickle(dir)
        return text

def make_embeddings(main_dir, name_data , text_col , cfg, EMBEDDING_TYPE):
    '''
    This function make all processes needed to make embeddings for the text column
    and encoding the scheme names
    Input:
        main_dir: the directory to the data
        name_data: The name of the data
        text_col: The name of text column we want to make embedding for
        cfg: config file

    Output:
        data_embeddings: embedded text data
        labels: encoded labels
    '''
    data = read_data(cfg.DATA_DIR , cfg.DATASET_NAME,type='observations')
    data_text = data[text_col] 
    file_dir = (f'{cfg.OUTPUT_DIR}/{name_data}_{EMBEDDING_TYPE}_tokenized.pkl' )
    if os.path.isfile(file_dir):
        tokenized_data = np.load(file_dir, allow_pickle=True)
    else:
        tokenized_data = tokenization(data_text)
        tokenized_data.to_pickle(file_dir)
    data_embeddings = text_encoder (tokenized_data,[] ,cfg )
    # Saving embeddings
    np.save(name_data+'_embeddings.npy' , data_embeddings)

    #Saving labels
    encoder = LabelEncoder()
    enc_path = ''
    if enc_path != '':
        encoder.classes_ = np.load(enc_path, allow_pickle= True)
    else:
        encoder.fit(data['observationscheme'])

    labels = encoder.transform(data ['observationscheme'])
    np.save('labels_' + name_data + '_embeddings.npy' , labels)
    return data_embeddings, labels

def read_embeddings(cfg):
    '''
    This function reads the embedding for the text data and encoded labels if they are made before
    '''
    data_name = cfg.EMBEDDING_DATA_NAME
    embd_path = os.path.join(cfg.EMBEDDING_DIR , data_name+'.npy')
    x_features = np.load(embd_path)
    label_name = 'labels_' + data_name + '.npy'
    embd_path = os.path.join(cfg.EMBEDDING_DIR , label_name)
    labels = np.load (embd_path)
    return  x_features, labels

def load_label_encoder (le, cfg):    
    '''
    This function read the encoded labels if they are made before
    '''
    enc_path =  cfg.LABEL_ENC_NAME +'.npy'
    le.classes_ = np.load (enc_path, allow_pickle= True)
    return le

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

if __name__ == "__main__":

    main_dir="" 
    proj_dir=os.path.join(main_dir,'/observation_classifier/models/observation_classification')    
    cfg_file = os.path.join(proj_dir,'laser_XGBoost.yml')
    data_dir = os.path.join(main_dir, '/all_data/v4')
    
    cfg_from_file(cfg_file)  

    cfg.DATA_DIR = data_dir
    cfg.EMBEDDING_DATA_NAME = ''
    cfg.LABEL_ENC_NAME = ''   
    cfg.OUTPUT_DIR = ''    
    cfg.DATASET_NAME = 'random_ObservationData_Combined_QA_comments'
    cfg.manualSeed = random.randint(1, 10000)
    cfg.EMBEDDING_DIR = ''
    random.seed(cfg.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPU_ID) 
    text_col = 'all_text'
    embedded_data, encoded_labels = make_embeddings(cfg.DATA_DIR, cfg.DATASET_NAME,text_col,cfg)