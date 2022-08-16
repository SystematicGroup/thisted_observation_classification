from sklearn.preprocessing import LabelEncoder
import numpy as np
import os


def encode_labels(data, col_name='obs_scheme', data_dir=''): 
    encoder = LabelEncoder()
    encoder.classes_ = np.load(f'{data_dir}/original_classes.npy', allow_pickle = True)
    data[col_name] = encoder.transform(data[col_name])
    return data

def subsample_label_embedder (data , col , dir, n): 
    '''
    Encodes the labels for different size of subsamples
    If the encoder for it is exists, then it just load it and transform the names to values
    If the encoder is not exist, first fit the label encoder to the data and then transform the labels
    '''
    name = 'classes_subsample_%d.npy' %n
    path = os.path.join(dir, name)
    le = LabelEncoder()

    if (os.path.exists(path)):
        le.classes_ = np.load(path, allow_pickle= True)
    else:     
        le.fit(data[col])    
    labels = le.transform(data[col])       
    np.save(path, le.classes_)
    
    return labels, le