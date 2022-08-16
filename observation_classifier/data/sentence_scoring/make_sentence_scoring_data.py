# Making the data ready for sentence scoring task 

from observation_classifier.data.pre_processing import *
from observation_classifier.data.pre_processing import PreProcessing
import pandas as pd
import pyarrow as pa
from datasets import Dataset
import random
random.seed(100)


class MakeSentScoringData():
    def __init__(self):
        self.pre_pro = PreProcessing()
    
    def make_comments(self, data, comments):
        ''''
        This function merge the data with comments based on ['myUnqualifiedId','unqualifiedversionid__'] keys
        and then convert the data in the format which is suitable for sentence scoring

        Input:
        ---------------------
            data: The input dataframe
            comments: The comments dataframe

        Output:
        ---------------------
            data_comments_cleaned (pd.DataFrame): Merged data from commnts and original data which some preprocessing is done for it
        
        '''
        merged_data_comments = self.pre_pro.merge_data_comments(data,comments)
        data_comments_cleaned = self.convert_comment_to_answer(merged_data_comments)
        
        return data_comments_cleaned

    def convert_comment_to_answer(self, data_comments):
        ''''
        This function convert comment dataframe in a right way for sentence scoring
            - replace 'comments_' column name to 'ObservationAnswer'
            - replace row values in 'ObservationQuestion' column to 'Bemærkning'

        Input:
        ---------------------
            data_comments: The input dataframe
        
        Output:
        ---------------------
            The cleaned dataframe of the comments data which is ready for sentence scoring
        '''
        # data_comments[['comments_']] = data_comments[['comments_']].replace(np.nan, '')
        data_comments_clean = data_comments.drop(['ObservationAnswer','ObservationQuestion','ObservationAnswer_len'], axis=1)
        data_comments_clean = data_comments_clean.rename(columns={'comments_': 'ObservationAnswer'})
        data_comments_clean['ObservationQuestion'] = 'Bemærkning'
        data_comments_clean[['ObservationAnswer']] = data_comments_clean[['ObservationAnswer']].replace(np.nan, '')
        return data_comments_clean

    def expand_splited_text(self, data, col_name):
        ''''
        This function expand the splitted text of each sample to the new row ----> for each sentence

        Input:
        ---------------------
            data: The input dataframe
            col_name: The name of the column we want to expand

        Output:
        ---------------------
            The expanded dataframe which the column contains just one sentence  

        '''
        df = data.apply(lambda x: pd.Series(x[col_name]),axis=1).stack().reset_index(level=1, drop=True)
        df.name = col_name    
        data = data.drop(col_name, axis=1).join(df)
        return data
    
    def make_pos_samples(self,data, params_dict):
        '''
        This function make all texts ready for Sentence scoring task
        It splits the whole text to sentences and make one sample for each sentence.
        It calculates lenght of the sentences to be ready for length filtering
        Does pre-processing: drop duplicatees
        All these data are from the original data and are positive samples and we assign label '1' to them.

        Input:
            - Data
        Output:
            - The splitted data by sentences in the text as ObservationAnswer and assigned positive labels
        '''
        assert type(data) == pd.DataFrame
        # Spliting answers    
        col_name = params_dict['col_name']
        data[col_name] = self.pre_pro.text_to_sent(data[col_name])
        
        # Expanding answers 
        data = self.expand_splited_text(data, col_name)
        data[col_name] = data[col_name].replace(np.nan, '')
        data = self.pre_pro.drop_duplicate(data ,subset=['ObservationQuestion','ObservationAnswer','ObservationScheme']) 
        data = self.pre_pro.len_sent(data,col_name, col_name + '_len')
        data = self.pre_pro.filter_number_of_tokens(data , col_name + '_len' , params_dict['token_threshold'])
        pos_data = self.pre_pro.token_to_string(data , col_name)
        pos_data['labels'] = 1
        return pos_data

    def make_neg_samples(self, data, change_col, scheme_col, label_col):
        ''''
        This function makes negative samples for the Q/A problem
        For each positive samples we generate a negative sample as the following way:
            - Replacing a wrong answer for each positive sample
            - What is a wrong answer?
                - A random answer from wrong samples
            - What are wrong samples for each positive sample?
                - They are all samples with other ObservationSchemes than the one the related positive sample has.
        Input:
        ---------------------
            data: The input dataframe
            change_col: The column we want to change to be wrong sample
            scheme_col: The name of the observation scheme
            label_col: The name of the column contains the binary labels

        Output:
        ---------------------
            The dataframe contains negative samples
        
        '''
        data = data.reset_index(drop=True)
        for index, row in data.iterrows(): 
            neg_samples = data[data[scheme_col] != row[scheme_col]] 
            selected_neg_sample = neg_samples.sample(random_state=100).iloc[0]
            row[change_col] = selected_neg_sample[change_col]
            row['ObservationAnswer_len'] = selected_neg_sample['ObservationAnswer_len']
            row[label_col] = 0         
            data.loc[index,:] = row
        return data

    def concat_two_data(self,df1,df2):
        ''''
        This function concatenate two dataframes

        Input:
        ---------------------
            df1: The first input dataframe
            df2: The second input dataframe

        Output:
        ---------------------
            The concatenated df
        
        '''
        pos_neg_data = pd.concat([df1,df2])
        pos_neg_data = pos_neg_data.reset_index(drop=True)
        sorted_pos_neg_data = self.pre_pro.sort_by_date(pos_neg_data,'LastUpdateDate',['year', 'month', 'day'])
        return sorted_pos_neg_data

    def find_combine_version(self, params_dict):
        '''
        This function retrieve the list of tokens which we should use for contenating diffrent text fields
        based on the concatenating version. 
        There are two different versions:
        - concatenating question, answer and observations fields
        - concatenating answer and observation fields

        We have used three different tokens to concatenate different fields which are '[SEP]' , '[CLS]' and ','.
        We use '[CLS]' token as a start token and we use '[SEP]' and ',' for between tokens.
        Based on 'combine-version' parameter, we use different list of tokens.
         '''

        combine_version = params_dict['combine_version']
        if params_dict['combine_type'] == 'SchemeAnswer':
            if combine_version == 0:
                split_token_list = ['','  ,']
            elif combine_version ==1:
                split_token_list  = ['', ' [SEP] ']
            elif combine_version==2:
                split_token_list  = ['[CLS] ', ' [SEP] ']

        elif params_dict['combine_type'] == 'SchemeQuestionAnswer':
            if combine_version == 0:
                split_token_list = ['', '  ,','  ,']
            elif combine_version ==1:
                split_token_list  = ['', ' [SEP] ', ' [SEP] ']
            elif combine_version==2:
                split_token_list  = ['[CLS] ', ' [SEP] ', ' [SEP] ']

        return split_token_list

    def combine_fields(self, data, params_dict):
        ''''
        This function combines a list of column inputs with a specific token, e.g. CLS, SEP or comma

        Input:
        ---------------------
            data: The input dataframe
            new_col_name: The name of the combined column in the data
            split_token: The special token which combine diffrent values of the different columns
            col_names_list: List of columns we want to combine

        Output:
        ---------------------
            The dataframe contains one additional column with combined values of columns

        '''
        new_col_name = params_dict['combine_type']
        split_token_list = self.find_combine_version(params_dict)
        if params_dict['combine_type'] == 'SchemeAnswer':
            col_names_list =['ObservationScheme', 'ObservationAnswer']
            data[new_col_name]= data.apply(lambda row: split_token_list[0] + row[col_names_list[0]]  + split_token_list[1] \
                                                        + row[col_names_list[1]],  axis = 1)

        elif params_dict['combine_type'] == 'SchemeQuestionAnswer': 
            col_names_list =['ObservationScheme', 'ObservationQuestion','ObservationAnswer']
            data[new_col_name]= data.apply(lambda row: split_token_list[0] +row[col_names_list[0]]  + split_token_list[1] \
                                                        + row[col_names_list[1]]+ split_token_list [2] + row[col_names_list[2]],  axis = 1)
        return data

    def convert_to_dataset(self,data, data_col_list, new_col_names):
        ''''
        This function converts the dataframe type of the data to Dataset format to be ready for training with transformers

        Input:
        ---------------------
            data: The input dataframe
            data_col_list: The list of columns we want to use for the dataset 

        Output:
        ---------------------
            The Dataset type which is made from the input dataframe

        '''

        data = data [data_col_list]
        data.columns = new_col_names
        data = Dataset(pa.Table.from_pandas(data))
        return data    

    def make_pos_neg_samples (self,data,params_dict):
        ''''
        This function do the following:
            - pre process pos data to be ready for sentence scoring
            - make negative samples based on the positive samples
            - concat positive and negative samples
            - based on the split version, concatenate columns

        Input:
        ---------------------
            data: The input dataframe
            col_name: The text column
            combine_version: The version we use for cancatenation of columns

        Output:
        ---------------------
            The cleaned data
        
        '''
        if params_dict['apply_superusers']:
            data = self.pre_pro.filter_superusers(data,params_dict['su_data_path'],params_dict['superusers_file_name'])
        pos_data = self.make_pos_samples(data, params_dict)
        save_data(pos_data, data_dir=params_dict['output_path'],name='ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_data.csv')

        #-----------------
        # making negative samples 
        #----------------- 
        neg_data = self.make_neg_samples(pos_data, 'ObservationAnswer', 'ObservationScheme', 'labels')
        save_data(neg_data, data_dir=params_dict['output_path'] ,name='ObservationData_cleaned_relavant_originalNames_preprocess_superusers_neg_data.csv')
        
        #-----------------
        # combining 
        #-----------------  
        #combining pos and neg data and shuffle them
        pos_neg_data = self.concat_two_data(pos_data,neg_data)
        save_data(neg_data, data_dir=params_dict['output_path'] ,name='ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_sorted.csv')

        return pos_neg_data
 
def do_making_pos_neg_data(params_dict,sent_scoring):
    """Make positive and negative samples
        - read the data and comments
        - make negative samples
        - concat pos and neg samples
        - save the final data to .csv file

    Args:
        params_dict (dict): Config file
        sent_scoring (_type_): Instance of MakeSentScoringData

    Output:
        all_data_and_comments: The concatenated positive and negative generated samples from the data answers and comments.

    """
    #-----------------
    # 1. Loading data
        # original data and comments
    #-----------------   
    data = read_csv_file(params_dict['data_path'],params_dict['data_file_name'])
    comments = read_csv_file(params_dict['data_path'], params_dict['comment_file_name'], sep = ';')

    #-----------------
    # 2. Making positive and negative data
    #-----------------       

    pos_neg_data = sent_scoring.make_pos_neg_samples(data,params_dict)

    #-----------------
    # 3. Making positive and negative comments
    #-----------------  
    comments = sent_scoring.make_comments(data,comments)
    pos_neg_comments = sent_scoring.make_pos_neg_samples(comments, params_dict)

    #-----------------
    # 4. Concatenating data and commets
    #-----------------    
    all_data_and_comments = sent_scoring.concat_two_data(pos_neg_data, pos_neg_comments)
    out_data_name = 'ObservationData_cleaned_relavant_originalNames_preprocess_superusers_pos_neg_data_comments_sorted.csv'

    if params_dict['save_data_flag']:
        save_data(all_data_and_comments,params_dict['output_path'],out_data_name)
    
    return all_data_and_comments

def do_const_split_data(pos_neg_data, params_dict, sent_scoring):    
    '''
    Activate if the parameter 'const_test_split_data' is True in config parameters
    This function split train and test sets in a way to use the constant test set for all kinds of experiments
    It uses 'const_num_test' last samples which are sorted by the date
    It saves the splitted data in two versions
        - csv: seperate files for train+val and test
        - pkl: a dictionary contains train+val and test by keys : ['train_val' , 'test']
    '''
    pos_neg_data = sent_scoring.pre_pro.sort_by_date(pos_neg_data,'LastUpdateDate',['year', 'month', 'day'])
    num_test = params_dict['const_num_test']
    splited_dataset = {}
    splited_dataset['test'] = pos_neg_data[-num_test:]
    splited_dataset['train_val'] = pos_neg_data[0:-num_test]
    name = params_dict['pos_neg_data_name'].split('.csv')[0]
    if params_dict['save_data_flag']:
        save_data(splited_dataset['test'],params_dict['output_path'],name + '_const_test.csv')
        save_data(splited_dataset['train_val'],params_dict['output_path'],name + '_const_train_val.csv')
        save_to_pickle(params_dict['output_path'], name + '_const_test_trainval_split.pkl' , splited_dataset)
    return splited_dataset

def do_combine_fields(params_dict):
    '''
    This function combines different fields and save the output.
    
    Inputs:
        - data: It can be splitted data (const-splitted data) or whole data
        - 'combine_type': The texts can be combined with different fields:
            1. SchemeAnswer: ObsrvationScheme +  ObservationAnswer
            2. SchemeQuestionAnswer: ObsrvationScheme + ObservationQuestion + ObservationAnswer
    Output:
        The data with combined fields 
    '''
    data_name = params_dict['const_split_data']
    data_type = data_name.split('.')[-1]
    combine_type = params_dict['combine_type']
    new_data_name = f'{combine_type}_{data_name}'
    if data_type =='csv':
        data = read_csv_file(params_dict['data_path'],params_dict['const_split_data'] )
        data = sent_scoring.combine_fields(data, params_dict)
        if params_dict['save_data_flag']:
            save_data(data,params_dict['output_path'],new_data_name)
    elif data_type == 'pkl':
        data = load_from_pickle(params_dict['data_path'], params_dict['const_split_data'])
        #combining Observation schemes and observation answers as a question
        for key in data.keys():
            data[key] = sent_scoring.combine_fields(data[key],params_dict)
        if params_dict['save_data_flag']:
            save_to_pickle(params_dict['output_path'],new_data_name, data)
    return

if __name__=="__main__":    
    
    PARAM_FILE_DIR = '/observation_classifier/data/sentence_scoring'       
            
    # Loading parameters from params file  
    sent_scoring = MakeSentScoringData()    
    try:
        param_file_name = f'make_sentence_scoring_data_config'
        params_dict = read_json_file(param_file_name,PARAM_FILE_DIR)
    except:
        print('The config does not exist make it by make_config_file.py')   

    if params_dict['const_test_split_data']:
        if params_dict['make_pos_neg_data']:       
            pos_neg_data = do_making_pos_neg_data(params_dict,sent_scoring) 
        else:
            try:
                pos_neg_data = read_csv_file(params_dict['data_path'],params_dict['pos_neg_data_name'])
            except:
                print('No file with given name can be loaded. Set the correct name of the file if you have made it before or set the parameter "make_sent_sco_data" to make the data')
        pos_neg_data = do_combine_fields(params_dict)
        trainval_test_dict = do_const_split_data(pos_neg_data,params_dict, sent_scoring)
    
    elif params_dict['make_pos_neg_data']:       
        pos_neg_data = do_making_pos_neg_data(params_dict,sent_scoring)   



    
    
  






   
    
    
    
    
