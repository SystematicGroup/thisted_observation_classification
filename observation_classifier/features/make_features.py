# This script make embeddings for the text column
import numpy as np
import torch
from laserembeddings import Laser

def text_embedder (text_data, encoder_name, encoder, n, mode, max_features=None): 
    
    if encoder_name== 'tfidf':           
        data = [(' '.join([str(elem) for elem in s])) for s in text_data]
        if mode == 'train':
            from sklearn.feature_extraction.text import TfidfVectorizer
            encoder = TfidfVectorizer(min_df= 5, max_df = 0.25, max_features=max_features)
            encoder = encoder.fit(data[0:n])
        x_features = encoder.transform(data)
        x_features = x_features.toarray().astype(np.float32)
    elif encoder_name == 'LaBSE' :
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('LaBSE')
        x_features = text_data.apply (lambda x: encoder.encode([(' '.join([str(elem) for elem in x]))]))
        x_features = np.vstack(x_features)   
    
    elif encoder_name == 'roberta-large':
        lens = [len(t) for t in text_data]
        max_len = max(lens)
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained(encoder_name)
        encoder = RobertaModel.from_pretrained(encoder_name) 
        x_features = text_data.apply (lambda x: torch.mean(encoder(**tokenizer([' '.join([str(elem) for elem in x])], return_tensors= "pt", max_length=512)).last_hidden_state [0] , 0).detach().numpy())
        x_features = np.vstack(x_features)
    
    elif encoder_name == 'laser':
        embedder = Laser()
        x_features = embedder.embed_sentences(text_data, lang='da') 

    return x_features