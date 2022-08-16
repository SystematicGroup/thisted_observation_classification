from operator import mod
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from observation_classifier.data.observation_classification.make_data import prepare_data
import pickle
import datetime
import dateutil.tz
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt 


def modelfit(model, x_train, y_train, x_val, y_val):
       '''
       Fit the model on the training data
       Input:
              model: The initial model: it can be a pretrained model or finetuned model
              x_train, y_train: Training samples
              x_val, y_val: Validation data
              cfg: Config file contains all parameters we need

       Output:
              model: It is a trained model. It can be pretrained model which is trained or
                     finetuned model which is trained with more epochs
       '''

       eval_set = [(x_train, y_train), (x_val, y_val)]
       
       #Loading a finetuned model
      
       model.fit(x_train, y_train, eval_set = eval_set , eval_metric="mlogloss", verbose=True )#,  xgb_model = our_model)
       # evaluating trained model on the validation data
       _ = evaluate (model , x_train, y_train , 'train')    
       return model 

def train(cfg):
       '''
       Making the data ready for training and validation
       Fitting the model on the data
       Evaluating the trained model 
       Input :
              cfg: Sll parameters are needed for training
       Output:
              Trained model
       '''
       # Making the training, validation and test data
       x_train, y_train, x_val, y_val, x_test, y_test, le = prepare_data(cfg)
       
       # Setting the parameters for the model
       params ={
              'learning_rate' : cfg.TRAIN.LR,
              'n_estimators' :  cfg.TRAIN.MAX_EPOCH,
              'gamma' : cfg.TRAIN.GAMMA,
              'use_label_encoder' : False,
              'objective' : 'multi:softmax', 
              'num_class' : len(le.classes_)-1,              
              'seed' : 2
       }  
       # If cuda is available do training on gpu
       if cfg.CUDA == True:
              params['tree_method'] = 'gpu_hist'
              params['gpu_id'] = cfg.GPU_ID
       xgbl = XGBClassifier()
       xgbl.set_params(**params)
       
       # Fitting the model to training data and evaluating on validation data
       model = modelfit(xgbl , x_train, y_train, x_val, y_val)

       save_model(model, cfg) # Saving the trained model

       _ = evaluate (model , x_test, y_test, 'test')        
       return model

def evaluate(model , x, y , mode = '', check_conf = True , check_topK= True , conf_list=None , k = 5):
       '''
       Evaluating the trained data on the data (val or test)
       It calculate model confidence with different values of confidence from 0 to 100%
       It calculates top-k accuracy from k=1 to K=5
       Input:
              model: Trained model
              x, y: The data for ecvaluating
              check_conf: If True do model_confidence
              check_topK: If True do top-k accuracy
              conf_list : List of confidence we are looking for
              k : The value of k we are looking for

       '''
       y_pred = model.predict(x)
       
       # Evaluate predictions and calculate accuracy
       accuracy = accuracy_score(y, y_pred)
       print("Accuracy of %s set: %.4f%%" % (mode , accuracy * 100.0))

       # Calculating accuracies with model confidence
       if check_conf:    
              print('The model confidence from %s for different values of confidence:' %(mode) )                                            
              y_pred_prob = model.predict_proba(x)
              ranked = np.argsort(y_pred_prob)
              tops = ranked[:,-1]
              if conf_list is None:
                     conf_list = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
              for conf in conf_list:
                     num_samples = []
                     ind= []                     
                     for idx , i  in enumerate ((y_pred_prob)):
                            if  i[tops[idx]]>conf :
                                   ind.append(idx)
                                   num_samples.append(i[tops[idx]])
                     
                     print(conf, len(num_samples) , accuracy_score(tops[ind], y[ind]))
       
       # Calculating accuracies with top-K
       if check_topK:
              print('The model accuracy for %s set with topK with different values of K:' %(mode))
              y_pred_prob = model.predict_proba(x)
              ranked = np.argsort(y_pred_prob)
              k = 5
              for n in range(k):              
                     top =[]
                     top = ranked[:,-n-1:]
                     true_num =0
                     for j in range (len(y_pred_prob)):
                            if np.isin(y[j],top[j] ):
                                   true_num = true_num+1
                     print(n+1 , true_num/len(y_pred_prob)*100)
       return accuracy

def save_model(model, cfg):
       ''''
       Saving the trained model
       Input:
              model: Trained model
              cfg: Parameters for saving the model
       '''

       now = datetime.datetime.now(dateutil.tz.tzlocal())
       timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
       stop_epoch = len(model.evals_result()[list(model.evals_result().keys())[0]] ['mlogloss'])
       dir = '%s/%s' % (cfg.OUTPUT_DIR, cfg.TEXT.EMBEDDING_MODEL)
     
       model_name = '%s.%s_%s_TR.%s.%d_DA.%s_LR.%.3f_gamma.%.3f_MaxEpoch.%d_stop.%d_%s.sav' % \
              (cfg.TEXT.EMBEDDING_MODEL , cfg.TEXT.MULTILINGUAL_TYPE , cfg.TRAIN.CLASSIFIER ,\
              cfg.TRAIN.USE_TRANSFER, cfg.TRAIN.TRANSFER_TYPE, cfg.AUG_DATA_NAME ,cfg.TRAIN.LR,\
              cfg.TRAIN.GAMMA, cfg.TRAIN.MAX_EPOCH , stop_epoch , timestamp) 
       if not os.path.isdir(dir):
              os.makedirs(dir)
       filename = os.path.join(dir,model_name)
       pickle.dump(model, open(filename, 'wb'))

def validate(cfg):
       ''''
       Using this function just for validation
       We can use the model which is trained in the script or just load a finetuned model
       The data can be standard data, or normal data
       Output:
              Accuracy, Top-K and model_confidence for input data
       '''
       try:
              loaded_model = pickle.load(open(cfg.TRAIN.MODEL, 'rb'))
       except:
              path = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.MODEL)
              loaded_model = pickle.load(open(path, 'rb'))

       if cfg.STD_VALIDATION == False:
              x_train, y_train, x_val, y_val, x_test, y_test,_ = prepare_data(cfg)
              _ = evaluate(loaded_model, x_train, y_train, mode='train')
              _ = evaluate(x_val, y_val, mode='validation')
              _ = evaluate(x_test, y_test, mode='test')

def plot_results(model, cfg):
       results = model.evals_result()
       epochs = len(results['validation_0']['mlogloss'])
       x_axis = range(0, epochs)
       stop_epoch = len(model.evals_result()[list(model.evals_result().keys())[0]] ['mlogloss'])
       dir = '%s/plots/%s' % (cfg.OUTPUT_DIR, cfg.TEXT.EMBEDDING_MODEL)
       title = '%s_TR.%s_LR.%.3f_gamma.%.3f_MaxEpoch.%d_stop.%d' % \
              (cfg.TRAIN.CLASSIFIER , cfg.TRAIN.USE_TRANSFER, cfg.TRAIN.LR,cfg.TRAIN.GAMMA,cfg.TRAIN.MAX_EPOCH , stop_epoch ) 
       
       # Plot log loss
       fig, ax = plt.subplots()
       ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
       ax.plot(x_axis, results['validation_1']['mlogloss'], label='Eval')
       ax.legend()
       plt.ylabel('Log Loss')
       plt.title(title)
       fig_name = '%s.png'  %(title)
       if not os.path.isdir(dir):
              os.makedirs(dir)
       fig_name = os.path.join(dir, fig_name)
         
       plt.savefig(fig_name)
