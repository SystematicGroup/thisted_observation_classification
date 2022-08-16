# Thisted Observation Classifier
Based on Systematic Data Science teams Project Template

The following components are used and included:

* Cookie-cutter for Data Science (empty folders are removed)
* FastAPI [Sebastián Ramírez]([)](https://github.com/tiangolo)
* Notebooks

* In the `notebooks` folder, create and/or run existing notebooks to analyse data.

## Requirements
Python 3.8+

## Installation
Install the required packages in your local environment
```bash
conda env create --name thisted_observation_classifier_env --file environment.yml
``` 

## Setup

```bash
python setup.py develop
```

--------

# Observation Classification
### Data
> How to make superusers:
- Create ```superusers.csv``` based on either the SU definition from Stine or based on statistics:
   - The first SU set was created as a list of good users from Stine. The code for filtering superusers will be done in ```/data/pre_processing.py``` with the function "filter_superusers"
   - The second is generated in ```notebooks/caretakerAnalysis.ipynb```. NB This notebook is messy, it takes place in the last part of the code.

> If you don't want to use superusers:
- In config_file ```data/sentence_scoring/make_config_file.py``` set the superuser flag to False.

> Preprocessing steps
1. Make config file for "pre_processing" from ```data/sentence_scoring/make_config_file.py``` with type "pre_processing".
   - Set the 'superuser_data' flag to True if the data should be filtered by superusers based on the list from Stine.
   - If the superusers should be based on statistics, run the notebook ```notebooks/caretakerAnalysis.ipynb``` before this step to create a list of superusers.
   Then set the flag to True and change the 'superusers_file_name' to match the new SU-list.
2. Preprocess the data. Run ```data/pre_processing.py```.
Data will be saved as *ObservationData_cleaned_relavant_originalNames_preprocess_superusers_filterLen.csv*.
3. Make the data with random combined for all text fields. Run ```data/observation_classification/combine_textcolumns_observations.py```.
Data will now be saved as *random_ObservationData_Combined_QA_comments.csv*.

> Embedding steps
4. Make sure to update the EMBEDDING_DIR in the configuration file for making embeddings: ```models/observation_classification/laser_XGBoost.yml```
5. Make embeddings. Run ```data/observation_classification/embeddings.py```. The name_data should be *random_ObservationData_Combined_QA_comments*. Embeddings will be saved in two files: 
- *random_ObservationData_Combined_QA_comments_embeddings.npy*
- *labels_random_ObservationData_Combined_QA_comments_embeddings.npy*


> Splitting
6. Splitting data will be done in the training phase below.

### Models
OBS: The BERT model was released in an internal Systematic repository. BERT training is thus not supported in this repository, but should be easy to develop. We therefore only document training from traditional ML models:

1. Training of XGBoost model:
   - Make sure to update the script file for training under ```models/observation_classification/laser.sh```, i.e. directories
   - Run ```models/observation_classification/laser.sh``` by ```bash laser.sh``` in terminal. It will use ```models/main.py ``` which calls ```models/train.py```'s function "train". This will split the data and fit the model.
2. Other traditional ML models:
   - Baseline models from sklearn can be found in ```models/observation_classification/baselines``` and in ```notebooks/observation_classification```


----------------
# Sentence Scoring
### Data
> Assuming we already ran the preprocessing steps above (1+2):
1. Make config file for pre_processing from ```data/sentence_scoring/make_config_file.py``` with type "make_sentence_scoring_data". 
2. Make the data. Run ```data/sentence_scoring/make_sentence_scoring_data.py```. This function will
   - Preprocess the data
   - Make positive and negative samples
   - Split the data
Data will be saved as *ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted_const_test_trainval_split.pkl*

3. Make embeddings. This will be a part of the training phase below.

### Models
Training
1. Make config file for training from ```data/sentence_scoring/make_config_file``` with conf_type "sentence_scoring_experiments".
2. Train the model. Run ```data/sentence_scoring/sent_score_train_exp.py```.
   - OBS: if you haven't tokenized the data, set tokenize_flag in config file to True.
   - Otherwise, it should have been run from ```data/sentence_scoring/DataTransformer.py``` and saved as three separate files.
3. Evaluate the model. 
   - Make config file for evaluation from ```data/sentence_scoring/make_config_file``` with conf_type "sentence_scoring_analysis".
   - Run ```data/sentence_scoring/sentence_scoring_evaluation.py``` to get metrics.
