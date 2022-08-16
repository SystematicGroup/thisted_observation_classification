from observation_classifier.data.observation_classification.embeddings import *
import sys

from observation_classifier.data.pre_processing import read_json_file

"""
1) Make a new .sh file in /models/observation_classification/experiments/new_observations called embedding_labse_newobs.sh
2) Make a new .yml file in  /models/cfg/multilingual
3) Run ./observation_classifier/models/observation_classification/experiments/new_observations/embedding_newobs.sh from root
4) Change EMBEDDING_TYPE name in the script below
"""

if __name__=="__main__":
    config = read_json_file(f"newobs_config",'/observation_classifier/data/observation_classification/new_observations')

    for typ in ['tfidf','roberta']:
        EMBEDDING_TYPE = typ

        # Setting directories 
        proj_dir = config['proj_dir']   
        
        if EMBEDDING_TYPE == 'tfidf':
            cfg_file = os.path.join(proj_dir,'cfg/tfidf/TFIDF_newobs.yml')
        if EMBEDDING_TYPE == 'LaBSE':
            cfg_file = os.path.join(proj_dir,'cfg/multi_lingual/LaBSE_newobs.yml')
        if EMBEDDING_TYPE == 'roberta':
            cfg_file = os.path.join(proj_dir,'cfg/multi_lingual/roberta_newobs.yml')

        data_dir = config['obs_data_dir']
        output_data_dir = os.path.join(data_dir, 'newobs/embeddings')
        
        # Load config file
        cfg_from_file(cfg_file)

        # Fill config args
        cfg.DATA_DIR = data_dir
        cfg.EMBEDDING_DATA_NAME = ''
        cfg.LABEL_ENC_NAME = ''   
        cfg.OUTPUT_DIR = output_data_dir   
        cfg.DATASET_NAME = 'ObservationData_Combined_QA_comments_superusers_lenmorethan5'
        cfg.manualSeed = random.randint(1, 10000)
        cfg.EMBEDDING_DIR = ''
        
        random.seed(cfg.manualSeed)
        if torch.cuda.is_available():
            torch.cuda.set_device(cfg.GPU_ID)
        
        # Run embeddings
        text_col = 'all_text'

        embedded_data, encoded_labels = make_embeddings(cfg.DATA_DIR, cfg.DATASET_NAME,text_col,cfg, EMBEDDING_TYPE)
        
        np.save(f'{cfg.OUTPUT_DIR}/{cfg.DATASET_NAME}_{EMBEDDING_TYPE}_embeddings.npy' , embedded_data)
        np.save(f'{cfg.OUTPUT_DIR}/labels_{cfg.DATASET_NAME}_{EMBEDDING_TYPE}_embeddings.npy' , encoded_labels)
