import os
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from observation_classifier.data.pre_processing import PreProcessing, save_data, read_json_file
from observation_classifier.data.observation_classification.random_combine_QA_observations import make_new_data


def preprocess_data(args, data_dir = '', save=False)-> pd.DataFrame:
    """Load raw data and process it:
        > Keep only goal observation schemes
        > Filter superusers
        > Merge text and comments
        > Combine the text fields to one column: 'all_text'
        > Filter out text fields with less than 5 tokens

    Args:
        data_dir (str): Path to data directory. Defaults to ''.
        args (_type_): Config
        save_data (bool): Save the data to csv. Defaults to False.
    """
    data_dir = args['obs_data_dir']
    # Raw data

    data = pd.read_csv('/all_data/v4/ObservationData_cleand.csv', sep= ';')
    df_com = pd.read_csv('/all_data/v4/ObservationComments_v4.csv', sep= ';')    
    pre_pro = PreProcessing()
    data = pre_pro.drop_date_numeric(data,['ObservationAnswer']) 
    # Observation types
    obs_scheme = data.copy().rename(columns={'ObservationScheme':'type'})
    goal_scheme = pd.read_csv(f'{data_dir}/Observation_types.csv', sep=';')

    # Keep relevant (goal) schemes
    
    data = pre_pro.keep_relevant_schemes(data, obs_scheme, goal_scheme).reset_index().drop(['index'],axis=1)

    #-----------------
    # Convert schemes to the original names
    #-----------------
    '''
    As we have done some pre processing to be able to fit the
    goal scheme file with our data (like removing spaces)
    Here we convert back the changed schemes to the original names
    '''
    logger.info("Converting schemes to the original names...")
    col = 'type'

    converter_path= "/data/npy_embeddings/sub_samples/classes"
    converter_name = 'convert_schemes_pattern.csv'
    decoded_schemes = data[col].to_frame()
    original_schemes = pre_pro.schemes_to_names (decoded_schemes,converter_path ,converter_name)
    data['ObservationScheme'] = original_schemes['original_scheme']
    df_combined = pre_pro.combine_answers_comments(data, df_com)
    
    # Filter superusers
    data_su = pre_pro.filter_superusers(
        df_combined,
        data_path='/interim/NewData_ranCombination/all_samples',
        superusers_name='superusers.csv'
    )

    # Random combination
    data_su = make_new_data(data=data_su, save_data=False)
        
    # Filter all_text column tokens less than 5
    df_len = pre_pro.len_text(data_su, col_name='all_text')
    df_len = pre_pro.filter_number_of_tokens(df_len, 'all_text', token_threshold=5)

    # Sort by date (newest at bottom)
    df_len_sort = pre_pro.sort_by_date(df_len,date_col='LastUpdateDate',sort_list= ['year', 'month', 'day'])
    if save:
        out_dir = args['obs_data_dir']
        save_data(df_len_sort, out_dir, 'ObservationData_Combined_QA_comments_normalCombine_superusers_lenmorethan5.csv')
    
    return data

def make_fixed_test_set(data:pd.DataFrame, args, save=False):
    """Generate fixed test set for New Observation-task

    Args:
        data (pd.DataFrame): Data to generate test set from. Intended for 'ObservationData_Combined_QA_comments_superusers_len>5.csv'
        args (dict): Config including argument "new_obs_data_dir" for saving the test set.
        save (bool, optional): Save test set. Defaults to False.

    Output:
        df_test (pd.DataFrame): Test set
    """
    test_size = int(np.round(len(data)*args['TEST_PERC']))
    df_test = data.iloc[-test_size:,:]

    if save:
        save_data(df_test, args['new_obs_data_dir'], name = "newobs_testset_fixed.csv")
    return df_test

def exclude_observation_schemes(data: pd.DataFrame, exclude_obsschemes):
    """Exclude chosen observation schemes from dataset

    Args:
        data (pd.DataFrame): Dataframe. Intended for 'ObservationData_Combined_QA_comments_superusers_len>5.csv' filtered by 'superusers.csv'
        obstypes ([str]): List of observation schemes that should be excluded from the dataset. Cleaned and concatenated version.

    Output:
        data_filtered (pd.DataFrame): Dataframe where obstypes has been filtered out
    """
    assert "observationscheme" in data
    data_filtered = data[~data.observationscheme.isin(exclude_obsschemes)]
    return data_filtered

def get_excluded_observation_schemes(data: pd.DataFrame, excluded_obsschemes):
    data_excl = data[data.observationscheme.isin(excluded_obsschemes)]
    return data_excl

def newobs_labelencoder(data, col_name='observationscheme', data_dir='', class_name = 'newobs_original_classes.npy'): 
    """Encode labels for observation classification task using Huggingface"""
    
    encoder = LabelEncoder()
    if os.path.isfile(os.path.join(data_dir,class_name)):
        encoder.classes_ = np.load(f'{data_dir}/{class_name}', allow_pickle = True)
    else:
        encoder  =  encoder.fit(data[col_name])
        np.save(os.path.join(data_dir,class_name), encoder.classes_)
     # this .npy file is made from this data: ObservationData_Combined_QA_comments_superusers_lenmorethan5.csv
    data[col_name] = encoder.transform(data[col_name])
    return data, encoder



if __name__=="__main__":

    # Load config file
    args = read_json_file(f"newobs_config",os.getcwd())

    #-------------------------------------
    # Load data
    #-------------------------------------
    if os.path.exists(f"{args['obs_data_dir']}/ObservationData_Combined_QA_comments_superusers_lenmorethan5.csv"):
        logger.info("Loading observation data")
        data = pd.read_csv(f"{args['obs_data_dir']}/ObservationData_Combined_QA_comments_superusers_lenmorethan5.csv")
    else:
        logger.info("'ObservationData_Combined_QA_comments_superusers_lenmorethan5.csv' does not exist, generating data")
        data = preprocess_data(args=args, save=False)


    # Load fixed test set
    if os.path.exists(f"{args['new_obs_data_dir']}/newobs_testset_fixed.csv"):
        logger.info("Loading fixed test set")
        test = pd.read_csv(f"{args['new_obs_data_dir']}/newobs_testset_fixed.csv")
    else:
        logger.info("'newobs_testset_fixed.csv' does not exist, generating testset")
        test = make_fixed_test_set(data, args, save=True)


    # Remove testset from data
    trainval = pd.merge(
        data,
        test,
        on = ['myUnqualifiedId','unqualifiedversionid__'],
        how = 'left',
        indicator=True,
        suffixes=('','_drop')
    ).query('_merge == "left_only"').drop(columns=['_merge'])
    trainval = trainval[[c for c in trainval.columns if not c.endswith('_drop')]]
    
    save_data(trainval, args['new_obs_data_dir'], "newobs_trainval_unfiltered.csv")

    #-------------------------------------
    # Exclude certain observation schemes
    #-------------------------------------
    exclude_obsschemes = ['blodsukkermåling','kontakttillæge']
    logger.info(f"Excluding observation schemes: {exclude_obsschemes}")
    trainval_filtered = exclude_observation_schemes(trainval, exclude_obsschemes)
    trainval_excl = get_excluded_observation_schemes(trainval, exclude_obsschemes)

    #-------------------------------------
    # Split train and val
    #-------------------------------------
    train, val = train_test_split(trainval_filtered, test_size=args['TEST_PERC'], random_state=42)
