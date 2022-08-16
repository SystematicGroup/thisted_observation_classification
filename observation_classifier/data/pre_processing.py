# This script does all preprocessing to make the data ready for different tasks
import pandas as pd
from loguru import logger
import torch
import spacy
import os
import json
import pickle
import numpy as np
import random
  
def make_json_file(params_dict, params_name, out_dir):
    file_path = os.path.join(out_dir,params_name + '.json')
    with open(file_path, 'w') as f:
        json.dump(params_dict, f)

def read_json_file(file_name, file_dir):
    file_path = os.path.join(file_dir,file_name + '.json')
    f = open(file_path)
    params_dict = json.load(f)
    return params_dict
    
def read_csv_file (file_dir, file_name, sep=''):
    file_path = os.path.join(file_dir, file_name)
    if sep == '':
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path, sep=sep)
    return data

def save_data (data, data_dir,name):
    path = os.path.join(data_dir, name)
    data.to_csv(path, index=False)

def save_to_pickle(out_dir, dataset_name , dataset):        
        out_path = os.path.join(out_dir, dataset_name)
        out_file = open(out_path, "wb")
        pickle.dump(dataset, out_file)
        out_file.close()

def load_from_pickle(out_dir, file_name ):        
        out_path = os.path.join(out_dir, file_name)
        out_file = open(out_path, "rb")
        out_file = pickle.load(out_file)       
        return out_file

def load_npy_file (file_dir, file_name):
    data_path = os.path.join(file_dir, file_name)
    data = np.load(data_path, allow_pickle=True)
    return data

class PreProcessing():
    def __init__(self) :
        pass
    def clean_text_column(self, df, column_name:str): 
        '''
        This function clean the Observation Scheme column to be ready for getting relavant goal schemes
        - It converts the whole text to the lower case
        - It replaces all the symbols and numbers with ''
        - It removes all spaces between charachters

        Input:
            - df: The data
            - column_name: The name of observation scheme column

        Output:
            - The data which its observation scheme column is cleaned 
        '''
        df[column_name] = df[column_name].str.lower()
        df[column_name] = df[column_name].str.replace('[^\w\s]','', regex=True)
        df[column_name] = df[column_name].str.replace('  ','')
        df[column_name] = df[column_name].str.replace(' ','')
        return df

    def keep_relevant_schemes(self, df_obs, observations_scheme, goal_scheme): 

        '''
        This function get those samples with observation schemes which are in out goal schemes from homecare unit.
        This function contains some pre-prcessing on observation schemes because the list we got as a goal schemes wasn't the same ones in the data.
        To be able to compare we have removed all spaces and symbols from both goal and data schemes and then keep the relevant schemes.
        The final data contains goal schemes without anu spaces and symbols.

        Input:
            - df: Our original data
            - column_name: List of goal observations
        
        Output:
            - Samples from the original data which their observation schemes included in the goal schemes

        '''
        logger.info("Only including relevant observation schemes")
        # Clean ObservationScheme column in ObservationData
        observations_scheme = self.clean_text_column(observations_scheme, 'type')

        # Clean column in the goal ObservationScheme df
        goal_scheme = self.clean_text_column( goal_scheme, 'type')

        # Keeping relevant ObsSchemes
        new_schemes = []
        for index , row in goal_scheme.iterrows():
            try:
                items =  df_obs[observations_scheme['type'] ==row['type']].iloc[0]
                com_schemes = (([row['type'],items['ObservationScheme']] ))
                new_schemes.append(com_schemes)
            except:
                print([row['type']])

        real_types = pd.DataFrame((new_schemes) , columns = ['merged_type','real_type'])
        real_types = real_types.drop(['merged_type'] , axis = 1)

        df1 = pd.DataFrame(df_obs['ObservationScheme'])
        df2 = pd.DataFrame(real_types['real_type'])
        # True ObservationSchemes
        observations = observations_scheme[df1.set_index(['ObservationScheme']).index.isin(df2.set_index(['real_type']).index)]
        
        return observations

    def sort_by_date(self,data, date_col, sort_list):
        '''
        This function sorts the input dataframe based on the date column
        Input:
        ---------------------------
            data: Input dataframe
            date_col: The name of the date column
            sort_list: List of columns for sorting if we have the same date

        Output:
        ---------------------------
            Sorted dataframe based on the date col and sort_list
        '''
    
        data[date_col] = pd.to_datetime(data[date_col], format="%Y/%m/%d")
        data['year'] = data[date_col].dt.year
        data['month'] = data[date_col].dt.month
        data['day'] = data[date_col].dt.day
        data = data.sort_values(sort_list,ascending=True)
        return data

    def filter_superusers(self,observations, data_path, superusers_name ):
        '''
        This function gets the samples from the list of users called superusers

        Input:
        -----------------------
            observations: The input dataframe
            data_path: The path to the file contains superusers names
            superusers_name: The name of the file contains the superusers names

        Output:
        -----------------------
            Output dataframe from superuses 

        '''
        superusers= read_csv_file(data_path, superusers_name)
        superusers_name = superusers.PractionerName
        superuser_data = observations[observations.PractionerName.isin(superusers_name)]
        return superuser_data

    def filter_number_of_tokens(self,observations , col_name , token_threshold):
        '''
        This function filters the samples based on the lenghts of the text field

        Input:
        ---------------------------
            observations: The input dataframe
            col_name: The column contains the lenghts of the text field we are filtering
            token_threshold: The threshold for filtering which is an integer number 

        Output:
        ---------------------------
            The dataframe which is filtered by the lenght of the text
        '''
        observations = observations[observations[col_name] >= token_threshold] 
        return observations

    def drop_nan(self, data,subset):
        ''''
        This function drops nan values from the list of columns

        Input:
        -------------------
            data: Input dataframe
            subset: The subset of column names we want to remove nan values from them

        Output:
        -------------------
            The dataframe which the nan values samples are removed 
        '''
        for col in subset:
            data = data[~(data[col].isnull())]
        return data

    def drop_duplicate(self,data,subset):
        ''''
        This function drops duplicates base on the subset of columns names

        Input:
        ---------------------
            data: Input dataframe
            subset: The subset of columns we want to apply to remove duplicates

        Output:
        ---------------------
            The dataframe which doesn't have any duplicates based on the list  of input subset columns

        '''
        data = data.drop_duplicates(subset=subset, keep='last')

        return data

    def drop_date_numeric(self,data, subset):
        ''''
        This function drops the samples has numeric values

        Input:
        ---------------------
            data: Input dataframe
            subset: The subset of columns we want to apply 

        Output:
        ---------------------
            The dataframe which doesn't have any numeric values in the input subset columns 

        '''
        for col in subset:
            data = data[~pd.to_datetime(data[col], errors='coerce').notnull()]
            data = data[~pd.to_numeric(data[col], errors='coerce').notnull()]
        return data

    def len_text(self,data, col_name):
        ''''
        This function calculates lenghts of the text

        Input:
        -----------------------
            - Input dataframe
            - The column contains tecxt
            - The new columns name which will contain the lenghts of the sentence

        Output:
        -----------------------
            - The output dataframe contains new column with the lenghts of the text column

        '''
        lens = data[col_name].str.split().apply(len)
        new_col = col_name + '_len'
        data[new_col] = lens
        return data

    def len_sent(self,data, col_name, col_len_name):
        ''''
        This function calculates lenghts of input sentence

        Input:
        -----------------------
            - data : Input dataframe
            - col_name : The column contains sentences
            - col_len_name : The new columns name which will contain the lenghts of the sentence

        Output:
        -----------------------
            - The output dataframe contains new column with the lenghts of the text column
        '''
        data[col_len_name] = data[col_name].apply(lambda x: len(x))
        return data

    def token_to_string(self,data,col_name):
        ''''
        This function combines the tokens of the input to be as a string sentence

        Input:
        ---------------------
            data: The input dataframe
            col_name: The column we are applying the function

        Output:
        ---------------------
            The dataframe contains string sentence instead of the list of the tokens

        '''
        data[col_name] = data[col_name].apply(lambda x: ' '.join(str(v) for v in x))
        return data
        
    def text_to_sent(self,data):    
        '''
        This function seprates the input free text to the sentences

        Input:
        ------------------------
            data: The column contains free text which we want to split

        Output:
        -----------------------
            The dataframe with splitted text to the sentences 
        '''
        if torch.cuda.is_available():
                    spacy.prefer_gpu()
        nlp = spacy.load("da_core_news_sm")
        doc = data.apply(nlp)
        sentences = doc.apply(lambda x: [sent for sent in x.sents])
        return sentences
    
    def schemes_to_names(self, decoded_schemes,converter_path, converter_name):

        ''''
        - This function converts cleaned schemes which we made to retrieve goal observation schemes to the original names which contains symbols like spaces and (,)

        Inputs:
            decoded_schemes: The observation scheme column from the data which is cleaned
            converter_path: The dir to the file for converting clened schemes to the original names
            converter_name: The name of the file which contains a path to convert cleaned schemes to the original names

        Outputs:
            A dataframe which is the converted from cleaned observation schemes to the original names
        '''
        # Loading the file containing converter names
        # Retrieving the original names from the file
        converter_pattern = read_csv_file(converter_path, converter_name)
        decoded_schemes = pd.DataFrame(decoded_schemes)
        decoded_schemes.columns = ['clean_scheme']
        merged_cleaned_schemes = decoded_schemes.merge(converter_pattern, how= 'left' , on=['clean_scheme'], sort=False)
        predicted_schemes_names = merged_cleaned_schemes ['original_scheme']
        predicted_schemes_names = pd.DataFrame(predicted_schemes_names)
        return predicted_schemes_names

    def shuffle_text(self,text):
        text = str(text)
        splitted_text = text.split('.')
        splitted_text.pop()
        random.shuffle(splitted_text)
        shuffled_text = '.'.join(splitted_text)

        return shuffled_text
        
    def make_new_data(self, data_path='', data=pd.DataFrame(), save_data = False):
        #---------------------------------------
        # 1. Reading the combined data of comments and answers
        #---------------------------------------
        if data.empty:
            data_name = 'ObservationData_Combined_QA_comments'
            data = read_data(data_path, data_name)

        #---------------------------------------
        # 2. shuffling the order of Joined observation data with comments
        #---------------------------------------
        logger.info("Shuffling the order of sentences in 'all_text' column ...")
        data['all_text'] = data['all_text'].apply(self.shuffle_text)


        #------------------------------------------
        # 3. Dropping duplicates and Nan values and save all new_data
        #------------------------------------------
        data = data.drop_duplicates(["all_text", "observationscheme"])
        data = data.dropna(subset = ["all_text"], inplace=True)

        if save_data:
            data.to_csv(f'{data_path}/random_ObservationData_Combined_QA_comments.csv')

        return data

    def merge_data_comments(self, data, comments):
        data_comments = pd.merge(
            data,
            comments,
            on=['myUnqualifiedId','unqualifiedversionid__']
        )
        return data_comments

    def combine_answers_comments(self, data, comments):
        '''
        We have different data for real the data and comments in the form of observations.
        Comments include just 'myUnqualifiedId','unqualifiedversionid__', 'comments_'.
        We need to join comments with the data to get all informations for the comments.
        We combine all answers and comments in one form Observation classification task. 
        We do some pre-processing like:
            - Drop-duplicates
            - Drop nan values
        Input:
            - data: The raw data with all columns
            - comments: The comments from the forms  and it contains just id, version and comments
        Output:
            - The concatenated data and comments which fields ObservationAnswer and comments_ are combined as one test names 'all_text'
        '''
        comments[['comments_']] = comments[['comments_']].replace(np.nan, '')
        joined_comments = comments.groupby(['myUnqualifiedId','unqualifiedversionid__'])['comments_'].apply('.'.join).reset_index()

        grouped_data = data.groupby(['myUnqualifiedId','unqualifiedversionid__'])
        joined_answers = data.groupby(['myUnqualifiedId','unqualifiedversionid__']).first().reset_index()
        joined_ans_in_group= grouped_data['ObservationAnswer'].apply('.'.join).reset_index()
        joined_answers['ObservationAnswer'] = joined_ans_in_group['ObservationAnswer']

        new_data = self.merge_data_comments (joined_answers, joined_comments)
  
        new_data['all_text'] = new_data['ObservationAnswer'].astype(str) + '.' + new_data['comments_']
        
        new_data[['all_text']] = new_data[['all_text']].replace(np.nan, '')
        new_data = self.drop_duplicate(new_data, subset=['ObservationComponentID','ObservationScheme', 'all_text'])
        new_data = new_data.drop(columns= ['ObservationQuestion'])
        new_data = self.drop_nan(new_data, subset=['all_text'])

        return new_data

    
def do_pre_processing(pre_params,p_pro):   
    '''
    -read the data
    -keep the data from the relavant (goal) schemes
    -retrieve original names of the schemes
    -do some pre-processing on the text data
    -remove duplicates
    -get superuser data if the flag is set
    -filter the data based on the number of tokens in the text column if the flag is set
    Input:
        params: Get parameters for pre_processing
    Output:
        Cleaned data and saving to .csv file
    '''
    #-----------------
    # 1. Loading the Original data
    #-----------------
    '''
    This data is the original data which some bugs in some rows are cleaned
    '''
    logger.info("Loading the original data...")
    data = read_csv_file(pre_params['data_path'], pre_params['data_name'], sep=';')
    
    #-----------------
    # 2. Keep relevant data
    #-----------------
    '''
    In this part we get thse samples which are from our goal schemes:
    goal_scheme: Is a file contains our goal schemes
    '''
    logger.info("Getting the relavant data...")
    obs_scheme = data.copy().rename(columns={'ObservationScheme':'type'})
    name = 'Observation_types.csv'
    goal_scheme = read_csv_file (pre_params['data_path'], name, sep=';')
    data = p_pro.keep_relevant_schemes(data, obs_scheme, goal_scheme).reset_index().drop(['index'],axis=1)
    if pre_params['save_data_flag']:
        save_data (data,pre_params['output_path'], 'ObservationData_cleaned_relavant.csv')
    # saved as : 'ObservationData_cleaned_relavant.csv'

    
    #-----------------
    # 3. Convert schemes to the original names
    #-----------------
    '''
    As we have done some pre processing to be able to fit the
    goal scheme file with our data (like removing spaces)
    here we convert back the changed schemes to the original names
    '''
    logger.info("Converting schemes to the original names...")
    col = 'type'
    decoded_schemes = data[col].to_frame()
    original_schemes = p_pro.schemes_to_names (decoded_schemes,pre_params['converter_path'] ,pre_params['converter_name'])
    #.to_frame().reset_index()
    data['ObservationScheme'] = original_schemes['original_scheme']
    if pre_params['save_data_flag']:
        save_data (data,pre_params['output_path'], 'ObservationData_cleaned_relavant_originalNames.csv')
    #saved_as : 'ObservationData_cleaned_relavant_originalNames.csv'
    
    #-----------------
    # 4. Doing text pre-processing  
    #-----------------
    '''
    In this part we do some pre-processing on the text fields
    1. Drop 'Nan' values
    2. Drop duplicates
    3. Calculate the lenght of the text column
    '''
    logger.info("Doing text pre-processing...")
    data = p_pro.drop_nan (data, ['ObservationAnswer'])
    data = p_pro.drop_duplicate(data ,['ObservationQuestion','ObservationAnswer', 'ObservationScheme'])  
    data = p_pro.drop_date_numeric (data, ['ObservationAnswer'])
    data = p_pro.len_text(data, 'ObservationAnswer')
    if pre_params['save_data_flag']:
        save_data (data, pre_params['output_path'], 'ObservationData_cleaned_relavant_originalNames_preproces_with_len.csv')

    #-----------------
    # 5. Filter superuser data
    #-----------------
    '''
    In this part we get the samples from superusers
    There is a file that contains the list of the superusers.
    We get the names from the file and get samples from the names
    '''
    if pre_params['superuser_data']:            
        logger.info("Getting data from superusers...")
        data = p_pro.filter_superusers (data, pre_params['data_path'], pre_params['superusers_file_name'])
        if pre_params['save_data_flag']:
            save_data (data, pre_params['output_path'], 'ObservationData_cleaned_relavant_originalNames_preprocess_superusers.csv')

    #-----------------
    # 6. Filter the data based on the lenghts of the text
    #-----------------
    '''
    In this part we filter the data based on the lenghts of the  text column
    There is a parameter named 'token_threshold' which shows the threshold
    '''
    if pre_params['filter_num_tokens']:
        logger.info("Filtering the data with the text lenght threshold...")
        data = p_pro.filter_number_of_tokens(data , 'ObservationAnswer_len' ,pre_params['token_threshold'])
        if pre_params['save_data_flag']:
            save_data (data, pre_params['output_path'], 'ObservationData_cleaned_relavant_originalNames_preprocess_superusers_filterLen.csv')
    return data


if __name__ == "__main__":
    params_file_dir = 'observation_classifier/data'
    try:
        pre_proc = True
        if pre_proc:
            param_file_name = 'pre_processing_config'
            pre_params = read_json_file(param_file_name,params_file_dir)
            p_pro = PreProcessing()
            do_pre_processing(pre_params,p_pro)
            # comments = pd.read_csv(f'{data_path}/raw/ObservationComments.csv', sep =';')

    except:
        print('The config does not exist make it by running make_config_file.py')

    

   
    