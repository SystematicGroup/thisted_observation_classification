import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from observation_classifier.data.plot_conf_matrix import save_plot
from observation_classifier.data.pre_processing import read_json_file, load_from_pickle
from observation_classifier.data.sentence_scoring.DataTransformer import DataTransformer
from observation_classifier.data.sentence_scoring.SentenceScoring import SentenceScoring
from observation_classifier.data.sentence_scoring.make_sentence_scoring_data import * 
from tensorflow.python.summary.summary_iterator import summary_iterator
from matplotlib import pyplot as plt
from collections import Counter

def load_models(params_dict):
    '''
    - makes an object of SentenceScoring
    - makes an object of DataTransformer
    - makes an object of MakeSentenceScoring
    - loads finetuned model
    '''
    ss = SentenceScoring(params_dict['checkpoint'])
    dt = DataTransformer(params_dict['tok_checkpoint'])
    mss = MakeSentScoringData()
    model = ss.load_model(params_dict['checkpoint'],params_dict['num_labels'])
    return ss, dt, mss, model

def get_ObsSchmes_list(raw_data):
    '''
    Retrieve raw data and the list of all ObservationSchemes to concatinate them to the test data for presicion@k
    '''
    schemes_list = pd.DataFrame(raw_data['train'].ObservationScheme.unique() , columns=['schemes'])
    return schemes_list

def make_evaluation_data(params_dict, test_data, dt, mss, split_token_list, schemes_list=[]):
    '''
    This function make the data ready for using the models from transformers
    - it does concatenation:
        - if it is 'precision@k: the answer of each sample concat to all ObservationScheme from the training data by seperator (to get all possibilities)
        - if it is 'eval_metric': the ObservationScheme concatenates to the related ObservationAnswer of the sample
    - it converts the pandas data to Dataset format 
    - it tokenizes the Dataset by related checkpoint
    
    Inputs:
        test_data: test data in dataframe format
        split_token_list: the list of splits we want to use. ex: splt tokens for sep_by_SEP
        schemes_list:
        - it gets list of all ObservationSchemes from training data for calculating precision@k
        - concates each samples's answer with all schemes in the list 
        
    Returns:
        - combined_data: the data contains the combined answer and observation by split tokens
        - combined_dataset: the combined dataset in the Dataset format
        - tokenized_data: the tokenized data with the related tokenizer from huggingface models
    '''
    if params_dict['combine_type'] == 'SchemeAnswer': 
        col_name = params_dict['col_name']
        if len(schemes_list) == 0 :
            combined_data = pd.DataFrame(split_token_list[0] + test_data['ObservationScheme'] + split_token_list[1]  + test_data[col_name], columns= ['text'])
            combined_data ['labels'] = test_data['labels']
        else:
            #
            combined_data = split_token_list[0] + schemes_list + split_token_list[1]  + test_data[col_name]
            combined_data.columns= ['text']
            combined_data['ObservationScheme'] = schemes_list #test_data['ObservationScheme']
            
    # Convert dataframe to Dataset format
    combined_dataset = mss.convert_to_dataset(combined_data, combined_data.columns , ['text','labels'])
    
    # Tokenize the dataframe with the related checkpoint
    tokenized_data = dt.tokenize(combined_dataset)

    return  combined_data, combined_dataset,tokenized_data
    
def precision_at_k(params_dict, model , pos_test, schemes_list, ss, dt, mss, split_token_list):
    '''
    This function calculate precision@k.
    The loaded finetuned model predicts the probability for each sample and sort them
        - predict probabilities for both pos and neg labels and save pos probs, because we are calculating precision
        - Sort the data by pos probabilities
        - Check if the scheme from the user is the subset of top-k possibilities
    '''
    # Making the data ready for evaluation
    combined_data, _, tokenized_data = make_evaluation_data(params_dict, pos_test, dt, mss, split_token_list, schemes_list=schemes_list)
    
    # Calculating probabilities
    probs = ss.get_predictions(model, tokenized_data).reshape(-1, 1)
    combined_data[['pos_prob']] = probs
    combined_data = combined_data.sort_values(by=['pos_prob'], ascending=False)
    
    if pos_test['ObservationScheme'] in set(combined_data['ObservationScheme'][0: params_dict['k']]):
        return 1
    else:
        return 0

def accuracy_with_conf(params_dict, prediction):
    '''
    This function calculates the model confidence with different values of confidence in the conf_list from config_file
    
    Input: 
        prediction: prediction from the model which contained the logits and real labels

    Returns:
        Calculates accuracy with diffrent values of confidence in the conf_list
    '''
    labels = prediction.label_ids # real labels
    probs = [softmax(i).tolist() for i in prediction.predictions] # calculates probabilities from the logits
    pred_labels = np.argmax(probs, axis=1) # #get labels
    probs = np.max(probs, axis=1) # gets probs for predicted labels
    for conf in params_dict['conf_list']:
        cond = probs>conf
        y_pred = pred_labels[cond]
        if len(y_pred) > 0:
            y_true = labels[cond]
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy with prediction_probability > {conf} : {accuracy}---> num_samples = {len(y_true)} from {len(labels)}")
        else:
            print(f"There is no samples with prediction_probability > {conf}")

def do_precisionK_analysis(params_dict, ss, dt, mss, model, raw_data, split_token_list):
    '''
    This function do all things needed for calculating precision@k analysis
    - gets list of all observations from the training data
    - get positive samples from the test data, because for precision we are focusing on positive samples
    - applies precision@k for all test samples
    - calculates the accuracy
    '''
    schemes_list = get_ObsSchmes_list(raw_data) # prepare list of all ObservationSchemes
    pos_test = raw_data['test'][raw_data['test'].labels==1] # get positive samples from the test data

    # For each test sample, concat with all observationSchemes and calculate top_k
    result = pos_test.apply(lambda x : precision_at_k(params_dict, model , x, schemes_list, ss, dt,mss, split_token_list), axis=1)
    accuracy = (sum(result.values)/(len(result))) *100
    print(f"precision@k : {accuracy}")
    return result, accuracy

def do_compute_metrics(params_dict,model,raw_data,ss,dt, mss,split_token_list):
    '''
    This function calculates 
    - all metrics are used during the training for the test data 
    - accuracy with confidence
    '''
    data_test = raw_data['test']

    # Making the data ready for evaluation
    _, _, tokenized_data = make_evaluation_data(params_dict, data_test, dt, mss, split_token_list)
    predictions = model.predict(tokenized_data) # predict logits from the finetuned model

    # Calculates accuracy with confidence
    accuracy_with_conf (params_dict , predictions)
    
    # calculates all metrics are used during the training for the test data 
    output = ss.compute_metrics([predictions[0],predictions[1]]) # send 'predictions' and 'label_ids' as the inputs 
    for metric in output.keys():
        print(f"{metric} : {output[metric]}")


    return output

def plot_train_process_analysis(values_dict, plot_list,labels_list, save_fig =True, save_dir = '', title=''):
    for i in range(len(plot_list[1])):
        plt.plot(values_dict[plot_list[0][0]],values_dict[plot_list[1][i]], label=plot_list[1][i])
    plt.legend()
    plt.xlabel(labels_list[0])
    plt.ylabel(labels_list[1])
    plt.title(title)
    if save_fig:
        save_plot(plt, save_dir, title)

def do_train_process_analysis(params_dict):
    '''
    THis function gets different values of parameters (ex:lr, epochs) and 
    values of calculated metrics (ex:train_loss, val_loss) from the log of the training process 
    '''
    log_path = os.path.join(params_dict['checkpoint'],params_dict['log_dir'], params_dict['log_name'])
    keys_list = ['epochs','train_loss', 'lr', 'val_loss', 'val_acc', 'val_f1','val_precision','val_recall','epochs']
    values_dict = {k: [] for k in keys_list}
    
    for summary in summary_iterator(log_path):
        for value in summary.summary.value:
            if value.tag == "train/epoch":
                values_dict['epochs'].append(value.simple_value)
            elif value.tag =="train/loss":
                values_dict['train_loss'].append(value.simple_value)
            elif value.tag == "train/learning_rate":
                values_dict['lr'].append(value.simple_value)
            elif value.tag =="eval/loss":
                values_dict['val_loss'].append(value.simple_value)        
            elif value.tag == "eval/accuracy":
                values_dict['val_acc'].append(value.simple_value)
            elif value.tag == "eval/precision":
                values_dict['val_precision'].append(value.simple_value)
            elif value.tag == "eval/recall":
                values_dict['val_recall'].append(value.simple_value)
            elif value.tag == "eval/f1" :
                values_dict['val_f1'].append(value.simple_value)
        
    # to have the same info for train and val sets, remove info for the first step, because we don't have values for validation set
    values_dict['train_loss'] = values_dict['train_loss'][1:]
    values_dict['lr'] = values_dict['lr'][1:]

    #because we have redundant values in the epochs
    values_dict['epochs'] = values_dict['epochs'][1:-2:2]
    return values_dict

def get_examples(ypred, ytrue, df, k):
    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []
    for i, element in enumerate(ypred):
        if element == ytrue[i]:
            if element == 1:
                true_pos.append(i)
            elif element == 0:
                true_neg.append(i)
        elif element != ytrue[i]:
            if element == 1:
                false_pos.append(i)
            elif element == 0:
                false_neg.append(i)

    true_pos_sample = random.choices(true_pos, k=k)
    true_neg_sample = random.choices(true_neg, k=k)
    false_pos_sample = random.choices(false_pos, k=k)
    false_neg_sample = random.choices(false_neg, k=k)

    print('True Positive: ')
    for i in true_pos_sample:
        print(df.iloc[i]['text'], i)
    print('\nTrue Negative:')
    for i in true_neg_sample:
        print(df.iloc[i]['text'], i)
    print(' \nFalse Positive:')
    for i in false_pos_sample:
        print(df.iloc[i]['text'], i)
    print(' \nFalse Negative:')
    for i in false_neg_sample:
        print(df.iloc[i]['text'], i)

    return true_pos, true_neg, false_pos, false_neg

def get_conf_schemes(test_dataset, true_pos, true_neg, false_pos, false_neg):
    scheme = []
    for element in test_dataset:
        scheme.append(element['text'].split('[SEP]')[0])

    unique_scheme = []
    list_set = set(scheme)
    # convert the set to the list
    unique_list = (list(list_set))
    for x in unique_list:
        unique_scheme.append(x)

    true_pos_schemes = []
    for element in true_pos:
        true_pos_schemes.append(scheme[element])
    true_pos_count = Counter(true_pos_schemes)

    false_neg_schemes = []
    for element in false_neg:
        false_neg_schemes.append(scheme[element])
    false_neg_count = Counter(false_neg_schemes)

    false_pos_schemes = []
    for element in false_pos:
        false_pos_schemes.append(scheme[element])
    false_pos_count = Counter(false_pos_schemes)

    true_neg_schemes = []
    for element in true_neg:
        true_neg_schemes.append(scheme[element])
    true_neg_count = Counter(true_neg_schemes)
    
    counts = [true_pos_count, false_neg_count, false_pos_count, true_neg_count]
    schemes = pd.DataFrame([true_pos_schemes, false_neg_schemes, false_pos_schemes, true_neg_schemes])
    schemes = schemes.transpose()
    schemes.columns=['true_pos_schemes', 'false_neg_schemes', 'false_pos_schemes', 'true_neg_schemes']

    return unique_scheme, counts, schemes

def bar_plot(results, xticks='int', save_fig =True, save_dir = '',title='TP_vs_FN',figsize =(25,15)):
    # set width of bar
    barWidth = 0.25
    fig,ax = plt.subplots(figsize =figsize)

    br1 = np.arange(len(results['scheme']))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, results['true_pos'],width = barWidth,label='True positive')
    plt.bar(br2,results['false_neg'],width = barWidth,label='False negative')

    if xticks=='int':
        idx = np.asarray([i for i in range(len(results['scheme']))])
        ax.set_xticks(idx)
    if xticks=='text':
        plt.xticks(br1,results['scheme'], rotation=90 )

    plt.legend()
    #plt.show()
    if save_fig:
        save_plot(plt, save_dir, title)

def scheme_plot_examples(results, loc, schemes, df, true_pos, false_neg, k):

    if type(loc) == str:
        results['scheme_no_space'] = results['scheme'].str.strip()
        loc = results.index[results['scheme_no_space'] == loc.strip()].tolist()[0]
    print(results['scheme'].iloc[loc])
    print('True positive: ', results['true_pos'].iloc[loc])
    print('False negative: ', results['false_neg'].iloc[loc])

    try:
        indexes_pos = [i for i,j in enumerate(schemes['true_pos_schemes']) if j == results['scheme'].iloc[loc]]
    except:
        pass
    try:
        indexes_neg = [i for i,j in enumerate(schemes['false_neg_schemes']) if j == results['scheme'].iloc[loc]]
    except:
        pass

    indexes_pos_sample = random.choices(indexes_pos, k=k)
    indexes_neg_sample = random.choices(indexes_neg, k=k)
    print('True positive: ')
    for element in indexes_pos_sample:
        print(df['text'].iloc[true_pos[element]])
    print('False Negative: ')
    for element in indexes_neg_sample:
        print(df['text'].iloc[false_neg[element]])
    
    return 


if __name__=="__main__":    
    PARAM_FILE_DIR = '/data/shared/thisted_observation_classification/observation_classifier/data/sentence_scoring'       

    try:
        param_file_name = f'sentence_scoring_analysis_config'
        params_dict = read_json_file(param_file_name,PARAM_FILE_DIR)
    except:
        print('The config does not exist, make it by make_config_file.py')   

    ss, dt, mss, model = load_models(params_dict)
    raw_data = load_from_pickle(params_dict['data_dir'], params_dict['data_name'])
    split_token_list = mss.find_combine_version(params_dict)  
    
    if 'precision_at_k' in params_dict['eval_list']:
        _ = do_precisionK_analysis(params_dict, ss,dt,mss,model,raw_data,split_token_list)
        
    if 'eval_metrics' in params_dict['eval_list']:
        _ = do_compute_metrics(params_dict,model,raw_data,ss, dt,mss,split_token_list)

    
    
    




