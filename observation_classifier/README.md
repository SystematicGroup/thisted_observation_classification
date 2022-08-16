# Folder structure
The repository has two main functionalities: *observation classification* and *sentence scoring*.

------------

**Generally, the src folder is divided into the following:**
```
observation_classifier
├── data
│   ├── observation_classification
│   │   └── new_observations
│   └── sentence_scoring
├── features
└── models
    ├── observation_classification
    └── sentence_scoring
```

**A detailed description of the folder structure:**
```
observation_classifier
├── data    
│   ├── encode_labels.py                            <- Label encoders for observation scheme
│   ├── get_accuracy_scores.py                      <- Performance measures
│   ├── pre_processing.py                           <- Pre process the raw data to be ready for different tasks  
│   ├── observation_classification            # Scripts focusing on Observation Classification (part 1 of the project)
│   │   ├── make_subsample_data.py                  <- make subsamples data with diffrent #samples and make text-embedding and encoded-labels
│   │   ├── combine_textcolumns_observations.py     <- Join two columns (answer and comments) together to "all_text"
│   │   ├── clean_standard_data.py                  <- Merge the data and comments
│   │   ├── embeddings.py                           <- Making embeddings for text column and encoding data labels
│   │   ├── load_model.py                           <- Load models for flat classification
│   │   ├── make_data.py                            <- Make data for observation classification
│   │   ├── make_subsample_data.py                  <- make subsample data ready for training for observation classification
│   │   ├── random_combine_QA_observations.py       <- Shuffling the order of sentences in combined (comments and answers) data
│   │   ├── user_analysis.py                        <- Analysis of specific users
│   │   ├── new_observations                  # Scripts focusing on researching the possibillity to adding new observations to Observation Classification
│   │   │   ├──make_config.py
│   │   │   ├──make_data.py
│   │   │   ├──Model.py
│   │   │   ├──newobsTrainer.py
│   │   └───└──split_embeddings.py       
│   ├── sentence_scoring                       # Scripts focusing on Sentence Scoring (part 2 of the project)
│   │   ├── DataTransformer.py                      <- Class for tokenizing data using transformer
│   │   ├── make_config_file.py                     <- Construct configuration file to run different scripts
│   │   ├── make_sentence_scoring_data.py           <- Make the data ready for sentence scoring task 
│   └── └── SentenceScoring.py                      <- Infrastructure for sentence scoring
├── features
│   ├── observation_classification
│   │   ├── new_observations
│   │   └──   └── make_embeddings.py
│   ├── __init__.py
│   └── make_features.py                            <- make embeddings for input text dataframe
├── models
│   ├── observation_classification              # Scripts focusing on Observation Classification (part 1 of the project)
│   │   ├── main.py                             <- main script for running different experiments for functional condition and health condition data
│   │   ├── train.py                            <- the script for training process for different experiment in classification tasks                      
│   │   ├── baselines
│   │   │   └── baselines_sklearn                       <- Baseline models using Sklearn
│   │   ├── miscc
│   │   │   ├── config.py                               <- general config file for observation classification
│   │   │   └── __init__.py
│   ├── sentence_scoring                        # Scripts focusing on Sentence Scoring (part 2 of the project)
└── └── └── sent_score_train_exp.py                 <- Fine-tuning a pretrained model


Updated: 27.06.2022
```