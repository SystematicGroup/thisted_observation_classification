"""
Baseline Models: Ridge Classifier and LDA for Observation Types
    - Imports
    - Functions
    - Load data
    - Setup same classes for train/val/test
    - Grid search
    - Fit model
    - Get results
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection, discriminant_analysis
from sklearn.metrics import accuracy_score
from loguru import logger
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for training a baseline model')
    parser.add_argument(
        '--n_jobs',
        dest='n_jobs',
        type=int,
        default=4
    )
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        default="ridge",
        choices=['ridge','lda']
    )

    args = parser.parse_args()
    return args


def get_model(model_name):
    if model_name == 'ridge':
        model = linear_model.RidgeClassifier(max_iter=1000)
    if model_name == 'lda':
        model = discriminant_analysis.LinearDiscriminantAnalysis()
    return model


def tuning_model(model_name, n_jobs=4):
    model = get_model(model_name)
    logger.info(f"Tuning model {model_name}")

    if model_name == 'ridge':
        params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'class_weight': [None, 'balanced'],
            'solver': ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            'tol': [0.0001, 0.001, 0.005, 0.01]
        }

    if model_name == 'lda':
        params = {
            'shrinkage': [None, 'auto', 0.001,0.02,0.4,0.6,0.8, 1.0],
            'solver': ['svd','lsqr','eigen'],
            'tol': [0.0001, 0.001, 0.005, 0.01]
        }
    grid = model_selection.GridSearchCV(
        model,
        param_grid=params,
        cv=3,
        verbose=3,
        n_jobs=n_jobs)
    return grid


def get_accuracy(model):

    logger.info(f"Predicting on train set, model {model}")
    ypred_train = model.predict(train[cols[:-1]])
    logger.info(f'Accuracy for train, model {model}: {accuracy_score(ypred_train, train[cols[-1]])}')
    
    logger.info(f"Predicting on val set, model {model}")
    ypred_val = model.predict(val[cols[:-1]])
    logger.info(f'Accuracy for val, model {model}: {accuracy_score(ypred_val, val[cols[-1]])}')
    
    logger.info(f"Predicting on unseen test set, model {model}")
    ypred_test = model.predict(test[cols[:-1]])
    logger.info(f'Accuracy for test set, model {model}: {accuracy_score(ypred_test, test[cols[-1]])}')

    return ypred_test


if __name__=='__main__':
    processed_data_dir = '/processed'
    args = parse_args()

    logger.info('>> Loading data')
    train = pd.read_csv(f'{processed_data_dir}/train_all.csv', compression='zip').drop(columns=['Unnamed: 0'])
    val = pd.read_csv(f'{processed_data_dir}/val_all.csv', compression='zip').drop(columns=['Unnamed: 0'])
    test = pd.read_csv(f'{processed_data_dir}/test_all.csv', compression='zip').drop(columns=['Unnamed: 0'])

    #------------------------------------------------------
    # Setup
    #------------------------------------------------------
    logger.info('>> Setup')
    tr_set = set(train['obs_scheme'].unique())
    v_set = set(val['obs_scheme'].unique())
    te_set = set(val['obs_scheme'].unique())
    t_list = list(tr_set.intersection(v_set).intersection(te_set))

    train = train[train.obs_scheme.isin(t_list)]
    val = val[val.obs_scheme.isin(t_list)]
    test = test[test.obs_scheme.isin(t_list)]

    cols = list(train.columns)

    #------------------------------------------------------
    # Model training
    #------------------------------------------------------
    logger.info('>> Model training and tuning')

    model_name = ['ridge','lda']
    
    # Train model
    model = get_model('ridge')
    model.fit(train[cols[:-1]], train[cols[-1]])

    model = get_model('lda')
    model.fit(train[cols[:-1]], train[cols[-1]])

    # Hyperparameter search and train
    grid = tuning_model(args.model, args.n_jobs)
        #grid = tuning_model(model_name[0])

    grid.fit(train[cols[:-1]], train[cols[-1]])
    logger.info(f"Best parameters for {grid}", grid.best_params_) 
    
    tuned = grid

    #------------------------------------------------------
    # Apply on test set
    #------------------------------------------------------
    logger.info('>> Accuracy scores')
    y_pred = get_accuracy(model)