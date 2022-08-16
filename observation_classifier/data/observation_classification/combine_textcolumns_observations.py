"""
This script will take the two text columns (answer and comments)
and join them together to one column = 'all_text'
"""

import pandas as pd
from observation_classifier.data.pre_processing import PreProcessing

if __name__=="__main__":

    # Load data
    pp = PreProcessing()

    data_path =''
    df_obs = pd.read_csv(f'{data_path}/raw/ObservationData_cleaned_relavant_originalNames_preprocess_superusers_filterLen.csv', sep = ';')
    observations_scheme = df_obs.copy().rename(columns={'ObservationScheme':'type'})
    df_com = pd.read_csv(f'{data_path}/raw/ObservationComments.csv', sep =';') # Raw, retrieved from Cura
    
    # Join observation data with comments and combine columns
    df = pp.merge_data_comments(df_obs, df_com)
    df_combined = pp.combine_answers_comments(df)
    df_combined_shuffled= pp.make_new_data(df_combined, save_data=True)
    