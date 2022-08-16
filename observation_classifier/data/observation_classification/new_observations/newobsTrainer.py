import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import importlib
from datetime import datetime
import numpy as np
from loguru import logger

from sklearn.model_selection import train_test_split, cross_val_score

from observation_classifier.data.sentence_scoring.DataTransformer import DataTransformer
from datasets import Dataset
import pyarrow as pa
from transformers import Trainer
from datasets import ClassLabel
import mlflow  

from observation_classifier.data.pre_processing import read_json_file
from observation_classifier.data.get_accuracy_scores import get_performance_scores
from observation_classifier.data.observation_classification.new_observations.Model import SklearnModel, HuggingfaceModel
from observation_classifier.data.observation_classification.new_observations.make_data import newobs_labelencoder

import warnings
warnings.filterwarnings('ignore')

class NewObservations():
    def __init__(self, model_type):
        self.model_type = model_type

    def add_obs_scheme(self, data: pd.DataFrame, scheme_data: pd.DataFrame, add_num: int):
        
        if add_num > len(scheme_data):
            print('Warning: add number bigger than number of schemes in training')
        #data_added = data.append(scheme_data.head(add_num), ignore_index=True)
        data_added = data.append(scheme_data.tail(add_num), ignore_index=True)
        data_added = data_added.sample(frac=1).reset_index(drop=True)

        return data_added

    def exclude_observation_schemes(self, data: pd.DataFrame, exclude_obsschemes):
        """Exclude chosen observation schemes from dataset

        Args:
            data (pd.DataFrame): Dataframe. Intended for 'ObservationData_Combined_QA_comments_superusers_len>5.csv' filtered by 'superusers.csv'
            obstypes ([str]): List of observation schemes that should be excluded from the dataset. Cleaned and concatenated version.

        Output:
            data_filtered (pd.DataFrame): Dataframe where obstypes has been filtered out
        """
        assert "observationscheme" in data, "Expected 'observationscheme' column in dataframe"
        data_filtered = data[~data.observationscheme.isin(exclude_obsschemes)]

        return data_filtered
    
    def get_excluded_observation_schemes(self, data: pd.DataFrame, excluded_obsschemes):
        """Get excluded chosen observation schemes from dataset

        Args:
            data (pd.DataFrame): Dataframe. Intended for 'ObservationData_Combined_QA_comments_superusers_len>5.csv' filtered by 'superusers.csv'
            obstypes ([str]): List of observation schemes that should be excluded from the dataset. Cleaned and concatenated version.

        Output:
            data_filtered (pd.DataFrame): Dataframe with only obstypes
        """
        data_excl = data[data.observationscheme.isin(excluded_obsschemes)]
        return data_excl
    
    def split_train_val_test(self, args, df_train_eval, df_test):
        """Split data into training, validation and test sets (X_features and y_labels)

        Args:
            df_train_eval (pd.DataFrame): Dataset incl. training set
            df_test (pd.DataFrame): Dataset for fixed test set

        Output:
            x_train, y_train, x_val, y_val, x_test, y_test: Features and labels for train, val and test set
        """

        if args['model_type'] == 'sklearn':
            X_train, X_val, y_train, y_val = train_test_split(df_train_eval.iloc[:,:-1], df_train_eval.iloc[:,-1:])
            X_test = df_test.iloc[:,:-1]
            y_test = df_test.iloc[:,-1:]
        
        if args['model_type'] == 'huggingface':
            pass
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    def do_crossvalidation(self, X_train, y_train, model, args):
        """Do cross-validation on train_eval set.

        Args:
            model : Fintuned model
            args : arguments for cross validation
        """
        scores = cross_val_score(model, X_train, y_train, cv=args['k_fold'])
        return scores

    def convert_excluded_scheme(self, exclude_obsschemes):
        """Convert from scheme name to embedded number

        Args:
            exclude_obsschemes (string): The name of the scheme to exclude

        Output:
            scheme (int): Embedded number for scheme
        """
        scheme = []
        if type(exclude_obsschemes[0]) == int:
            pass
        elif type(exclude_obsschemes[0]) == str:
            dat = np.load('/newobs_original_classes.npy', allow_pickle=True)
            scheme.append(np.where(dat == exclude_obsschemes[0])[0][0])
        return scheme

    def plot_stats(self, data,args, numbers,title:str):
        """Unzip and plot the statistics for a given dataset

        Args:
            data (list[list]): The data to unzip and plot, list of lists
            title (string): The name and title of the plot

        Output:
            
        """
        acc, f1score, precision, recall = zip(*data)

        fig, ([ax1, ax2],[ax3, ax4]) = plt.subplots(2, 2, sharex=True, figsize=(25,15))
        fig.suptitle(f'{title} - Last sample is with full dataset')
        ax1.plot(numbers,acc)
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Number of samples')
        ax2.plot(numbers,f1score)
        ax2.set_title('F1 Score')
        ax2.set_xlabel('Number of samples')
        ax3.plot(numbers,precision)
        ax3.set_title('Precision')
        ax3.set_xlabel('Number of samples')
        ax4.plot(numbers,recall)
        ax4.set_title('Recall')
        ax4.set_xlabel('Number of samples')
        today = datetime.now()
        d1 = today.strftime("%d_%m_%Y_%H:%M")
        fig_name =  f'{title}_{d1}.png'
        fig_path = os.path.join(args['checkpoint_dir'],fig_name)
        plt.savefig(fig_path)

    def make_hf_data(self, data, dt):
        '''
        This function make the data ready to be used by huggingface models.
        1. We retrieve the input text and labels from the data
        2. We convert the data to the Dataset format
        3. We tokenize the data by using the related tokenizer from huggingface
        '''
        data = data.rename(columns={'observationscheme':'label','all_text':'text'})
        data = Dataset(pa.Table.from_pandas(data[['text','label']]))
        data = dt.tokenize(data)   
        return data

def retrain_hugging_model(traineval, args, train_args, dt, model= None, include_obs = False):
    '''
    This function retrain the huggingface model with/without the new observaion
    If the 'include_obs' parameter is true:
        - The function loads the trained model without the new observation 
        - Change the last layer of the model to work with the new size of the labels
    Otherwise:
        - The function upload the specific huggingface model.
    - We split the data to train and validation sets
    - We convert the data to the format which is acceptable for huggingface models
    - We train the huggingface model with training set and evaluate the model after each epoch with the validation set 
    '''
    num_labels = traineval.observationscheme.nunique()
    if include_obs:     
        model.load_model(args['checkpoint_dir'],args['excluded_checkpoint'],num_labels=num_labels, trainer=False)
    else :
        model.set_model(args['checkpoint'])
        model.load_hf_model(num_labels=traineval[args['col_name']].nunique())
    
    train, validation = train_test_split(traineval, test_size=0.2)
    train_dataset = new_obs.make_hf_data(train, dt)
    val_dataset = new_obs.make_hf_data(validation, dt)    
    trainer = model.train(args, train_dataset, val_dataset, train_args=train_args)   
    return trainer

def eval_save_finetuned_model(args, test_dataset, model, trainer, num_included_samples = 0, include_obs=True, excluded_dataset= None):
    logger.info(">> Evaluating model:")  
    output = model.predict(test_dataset,trainer)
    logger.info(f"Prediction metrics: {output.metrics}")
    if excluded_dataset:
        excluded_output = model.predict(excluded_dataset,trainer)
        logger.info(f"Prediction metrics for excluded data with size {num_included_samples} samples: {excluded_output.metrics}")
    today = datetime.now()
    d1 = today.strftime("%d_%m_%Y_%H:%M")
    excluded_obs = args['obsscheme_exclude_name']
    if not include_obs:      
        model_name = f'finetuned_model_excluded_obs'
        for obs in excluded_obs:
            model_name = model_name + '-' + obs 
        model_name = model_name + f'_{d1}'
        mlflow.end_run()
        mlflow.start_run(nested=True)
        args['excluded_checkpoint'] = model_name
    else:
        model_name = f'finetuned_model_included_obs'
        for obs in excluded_obs:
            model_name = model_name + '-' + obs
        model_name = model_name + f'_with_{num_included_samples}-samples_{d1}'
    model.save_model(trainer , args['checkpoint_dir'], model_name)
    if excluded_dataset:
        return output.metrics, excluded_output.metrics
    else:
        return output.metrics

def load_fintuned_model(data, model_dir, model_name, model):
    num_labels = data.observationscheme.nunique()
    trainer = model.load_model(model_dir, model_name, num_labels= num_labels, trainer = True)

    return trainer 

if __name__=='__main__':

    config_dir = '/observation_classifier/data/observation_classification/new_observations'
    args = read_json_file(f"newobs_config",config_dir)

    if args['model_type'] == 'sklearn':
        model = SklearnModel()

        index = args["sklearn_model_name"].rfind(".")
        model_name = args['sklearn_model_name'][:index]
        classname = args["sklearn_model_name"][index+1:]

        m = importlib.import_module(model_name)
        model_attr = getattr(m, classname)
        model_class = model_attr(**args['param_dict'])

    if args['model_type'] == 'huggingface':
        model = HuggingfaceModel()

    new_obs = NewObservations(args['model_type'])
    
    #--------------------------------------
    # Sklearn
    #--------------------------------------
    if args['model_type'] == 'sklearn':
        # Load embedded data
        df_train_eval = pd.read_csv(f"{args['emb_data_dir']}/newobs_{args['embed_type']}_train_eval.csv").rename(columns={'labels':'observationscheme'})
        df_test = pd.read_csv(f"{args['emb_data_dir']}/newobs_{args['embed_type']}_test_fixed.csv").rename(columns={'labels':'observationscheme'})

        # Remove specific obs_schemes
        obsscheme_excl = args['obsscheme_exclude']
        obsscheme_excl = new_obs.convert_excluded_scheme(obsscheme_excl)
        added_num = args['added_num']
        df_train_eval_filtered = new_obs.exclude_observation_schemes(df_train_eval, obsscheme_excl)
        df_train_eval_excluded = new_obs.get_excluded_observation_schemes(df_train_eval, obsscheme_excl)


        single = []
        all = []
        numbers = []

        for num in range(0,100, 5):
            if num > len(df_train_eval_excluded):
                break
            numbers.append(num)
            df_added_data = new_obs.add_obs_scheme(df_train_eval_filtered,df_train_eval_excluded, num)
            
            # Split
            X_train, y_train, X_val, y_val, X_test, y_test = new_obs.split_train_val_test(args, df_added_data, df_test)
            # Train
            model.set_model(model_class)
            model.train(X_train, y_train)


            # Predict
            y_pred = model.predict(X_test)
            print('All classes:')
            all.append(get_performance_scores(y_test, y_pred))
            
            # Confusion matrix
            fig, ax = plt.subplots(figsize=(25,15))
            cm=pd.crosstab(y_test.values.flatten(), y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt=''
            )

            df_test_excluded = new_obs.get_excluded_observation_schemes(df_test, obsscheme_excl)
            X_test_excluded = df_test_excluded.iloc[:,:-1]
            y_test_excluded = df_test_excluded.iloc[:,-1:]
            y_pred_excluded = model.predict(X_test_excluded)

            y_test_bin = [1] * len(y_test_excluded)
            y_pred_bin = []
            for val in y_pred_excluded:
                if val == obsscheme_excl:
                    y_pred_bin.append(1)
                else:
                    y_pred_bin.append(0)

            print('\nSingle performance score:')
            single.append(get_performance_scores(y_test_bin, y_pred_bin))

        print('\nFull dataset:')
        numbers.append(num+10)
        X_train, y_train, X_val, y_val, X_test, y_test = new_obs.split_train_val_test(args, df_train_eval, df_test)

        model.set_model(model_class)
        model.train(X_train, y_train)

        y_pred_excluded = model.predict(X_test_excluded)

        y_test_bin = [1] * len(y_pred_excluded)
        y_pred_bin = []

        for val in y_pred_excluded:
            if val == obsscheme_excl:
                y_pred_bin.append(1)
            else:
                y_pred_bin.append(0)

        y_pred = model.predict(X_test)
        print('All classes:')
        all.append(get_performance_scores(y_test, y_pred))
        print('\nSingle performance score:')
        single.append(get_performance_scores(y_test_bin, y_pred_bin))

        new_obs.plot_stats(all,numbers, 'all')
        new_obs.plot_stats(single,numbers, 'single')

    #--------------------------------------
    # Huggingface
    #--------------------------------------
    if args['model_type'] == 'huggingface':
        obsscheme_excl = args['obsscheme_exclude_name']
        # Loading data and transformers
        traineval = pd.read_csv(os.path.join(args['new_obs_data_dir'], args['new_obs_trainval_data']))        
        test = pd.read_csv(os.path.join(args['new_obs_data_dir'], args['new_obs_test_data']))
        dt = DataTransformer(checkpoint=args["checkpoint"])
        # Filtering
        #Filtering the train/val/test data from the excluded data
        # Seperating the excluded observation data 
        traineval_filtered =  new_obs.exclude_observation_schemes(traineval, obsscheme_excl)
        traineval_excluded = new_obs.get_excluded_observation_schemes(traineval, obsscheme_excl)
        test_filtered = new_obs.exclude_observation_schemes(test, obsscheme_excl)
        test_excluded = new_obs.get_excluded_observation_schemes(test, obsscheme_excl)
        
        # Encoding labels
        '''
        There are two different strategy for retraining huggingface models:
        version 1:
            - We gues that the data has all observations and we add a fake sample for the observation we want to exclude.
            - We use sklearn label encoder (num_labels = num(all_obs) )
            - We train the model without the excluded observation
            - We add excluded observation to the data and retrain the model.        
        version 2:
            - We filter the data from the observation we want to exclude
            - We use 'ClassLabel' property from the huggingface package as a label encoder (num_labels = num(all_obs) -1 )
            - We train the model without excluded observation
            - We add the excluded observation to the data
            - We add the new observation to the ClassLabel
            - We retrain the model by changing the last layer to work with the new number of observation
                (using property ignore_mismatched_sizes=True)
        ''' 
        excluded_test = []
        all_test = []
        if args['hf_training_version'] == 2:   
            label_names = traineval_filtered.observationscheme.unique() # Get the name of the classes in the data
            labels = ClassLabel(names=label_names) # Make the label encoder 
            traineval_filtered.observationscheme = labels.str2int(traineval_filtered.observationscheme) # Encode train labels using the label encoder
            test_filtered.observationscheme = labels.str2int(test_filtered.observationscheme) # Encode test labels using the label encoder
            new_label_names = np.append(label_names , obsscheme_excl ) # Adding the new obs label
            new_labels = ClassLabel(names=new_label_names) # Make a new label encoder with the new observation
            traineval_excluded.observationscheme =new_labels.str2int(traineval_excluded.observationscheme) # Convert train labels using the new encoder with new obs
            test_excluded.observationscheme = new_labels.str2int(test_excluded.observationscheme) # Convert excluded test labels using the new encoder with new obs
            test.observationscheme = new_labels.str2int(test.observationscheme) # Convert test labels using the new encoder with new obs
        
        elif args['hf_training_version'] == 1:            
            traineval_filtered, traineval_encoder = newobs_labelencoder(traineval_filtered, data_dir= args['new_obs_original_classes_dir'], class_name=args['new_obs_original_classes'])
            traineval_excluded, _ = newobs_labelencoder(traineval_excluded,data_dir= args['new_obs_original_classes_dir'], class_name=args['new_obs_original_classes'])
            test, _ = newobs_labelencoder(test,data_dir= args['new_obs_original_classes_dir'], class_name=args['new_obs_original_classes'])
            test_excluded,_ = newobs_labelencoder(test_excluded,data_dir= args['new_obs_original_classes_dir'], class_name=args['new_obs_original_classes'])
            test_filtered,_ = newobs_labelencoder(test_filtered,data_dir= args['new_obs_original_classes_dir'], class_name=args['new_obs_original_classes'])      
            fake_sample = traineval_excluded.iloc[0] 
            fake_sample['all_text'] = 'Dette er en eksempel tekst.'
            traineval_filtered = traineval_filtered.append(fake_sample, ignore_index=True)
        today = datetime.now()
        d1 = today.strftime("%d_%m_%Y_%H-%M")
        # Making the filtered test 
        test_filtered_dataset = new_obs.make_hf_data(test_filtered, dt)
        model_excluded  = retrain_hugging_model(traineval_filtered, args, args['excluded_training_param_dict'],dt, model=model) #load_fintuned_model(traineval_filtered, model_dir, model_name=model_name,model=model) #
        out_metrics = eval_save_finetuned_model(args, test_filtered_dataset, model, model_excluded, include_obs=False)

        name = f'excluded_obs_{obsscheme_excl[0]}_results_{d1}.json'
        out_path =  os.path.join(args["checkpoint_dir"], name)
        with open(out_path, 'w') as fout: #Saving the results of test evaluation/not included new obs
                fout.write(json.dumps(out_metrics, indent=4))
        
        test_excluded_dataset = new_obs.make_hf_data(test_excluded, dt)
        test_dataset = new_obs.make_hf_data(test, dt)
        nums = []      
        all_nums = [30, 60, 100, 200, len(traineval_excluded)]        
        for added_num in all_nums:
            nums.append(added_num)
            df_added_data = new_obs.add_obs_scheme(traineval_filtered,traineval_excluded, added_num)  
            model_included  = retrain_hugging_model(df_added_data, args, args['included_training_param_dict'], \
                    dt, model=model, include_obs=True)
            out_metrics , excluded_out_metrics= eval_save_finetuned_model(args, test_dataset, model, model_included, num_included_samples=added_num, \
                    include_obs=True, excluded_dataset=test_excluded_dataset)
            all_test.append(out_metrics)
            excluded_test.append(excluded_out_metrics)
                
        data = [nums, all_test, excluded_test,nums]
        names = [f'excluded_obs_{obsscheme_excl[0]}_num_of_samples_results_{d1}.json',f'excluded_obs_{obsscheme_excl[0]}_all_test_results_{d1}.json',f'excluded_obs_{obsscheme_excl[0]}_excluded_test_results_{d1}.json']
        for i in range(len(names)):
            out_path =  os.path.join(args["checkpoint_dir"], names[i])
            with open(out_path, 'w') as fout:
                fout.write(json.dumps(data[i], indent=4))
        keys_to_extract = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']

        all_test_exctracted = []
        excluded_test_extracted = []

        for test in all_test:
            all_test_exctracted.append([test[key] for key in keys_to_extract])
        
        for test in excluded_test:
            excluded_test_extracted.append([test[key] for key in keys_to_extract])
        
        new_obs.plot_stats(all_test_exctracted,args,nums ,'all')
        new_obs.plot_stats(excluded_test_extracted, args, nums, 'single')