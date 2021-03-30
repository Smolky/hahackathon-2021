"""
    Keras and Talos por hyper-parameter tunning
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

# Load basic stuff
import random
import os
import glob
import tensorflow
import talos
import datetime
import sys
import csv
import pickle
import argparse
import config
import time
import sklearn
import numpy as np
import pandas as pd
import pkbar
import utils
import kerasModel

from tensorflow import keras
from keras import backend as K
from datasetResolver import DatasetResolver
from preprocessText import PreProcessText



# Parser
parser = argparse.ArgumentParser (description = 'Retrieve a model from Keras')
parser.add_argument ('--dataset', dest = 'dataset', default = "hahackathon", help="|".join (config.datasets.keys ()))
parser.add_argument ('--force', dest = 'force', default = False, help="If True, it forces to replace existing files")
parser.add_argument ('--minutes', dest = 'minutes', default = 60 * 24 * 7, type = int, help = "Number of limits for the evaluation. Default is one week")
parser.add_argument ('--task', dest = 'task', default = "1a", help = "Get the task")



# Parser
args = parser.parse_args ()


# @var max_words Tokenizer max words
max_words = None


# Remove previous CSVs and data
if os.path.exists (args.dataset + "_model.zip"):
    os.remove (args.dataset + "_model.zip")
    
files = glob.glob (args.dataset + '/*.csv')
for f in files:
    os.remove(f)
    
files = glob.glob (args.dataset + '_model/*')
for f in files:
    os.remove(f)


if os.path.exists (args.dataset):
    os.rmdir (args.dataset)

if os.path.exists (args.dataset + '_model'):
    os.rmdir (args.dataset + '_model')


# Preprocess text
preprocess = PreProcessText ()


# @var umucorpus_ids int|string The Corpus IDs
for key, dataset_options in config.datasets[args.dataset].items ():
    
    # Resolver
    resolver = DatasetResolver ()


    # Get the dataset name
    dataset_name = args.dataset + "-" + key + '.csv'


    # Get the dataset
    dataset = resolver.get (dataset_name, dataset_options, args.force)


    # Determine the task type
    task_type = 'classification' if args.task in ['1a', '1c'] else 'regression'


    # Get linguistic features
    df_lf = pd.read_csv (os.path.join (config.directories['assets'], args.dataset, key, 'lf.csv'), header = 0, sep = ",")
    df_lf = df_lf.rename (columns = {"class": "label"})
    df_lf = df_lf.loc[:, (df_lf != 0).any (axis = 0)]
    

    # Get the dataset as a dataframe
    df_embeddings = dataset.get ()
    df_embeddings = df_embeddings.drop (df_embeddings[df_embeddings['is_test'] == True].index)
    df_embeddings = dataset.getDFFromTask (args.task, df_embeddings)
    
    
    # Get the same LF that the embedding layer
    df_lf = df_lf.head (df_embeddings.shape[0])
    
    
    # Perform feature selection over the LF
    reg = sklearn.linear_model.LassoCV (normalize = True)
    
    
    # Get all features that are not the label
    X = df_lf.loc[:, ~df_lf.columns.isin (['label'])]
    
    
    # Encode label
    # @todo. Error with multiclass
    y = df_lf['label'].astype ('category').cat.codes
    
    
    # Fit LassoCV
    reg.fit (X, y)
    
    
    # Get Lasso coefficients
    coef = pd.Series (reg.coef_, index = X.columns)
    
    
    # Determine which LF does not fit the coef
    lf_columns_to_drop = [colulmn for colulmn, value in coef.items () if value == 0]
    
    
    # Filter those LF
    df_lf = df_lf.drop (columns = lf_columns_to_drop)
    
    
    # Preprocess word embeddings. First, some basic stuff
    for pipe in ['remove_urls', 'remove_digits', 'remove_whitespaces', 'remove_elongations', 'to_lower', 'remove_punctuation']:
        df_embeddings['tweet'] = getattr (preprocess, pipe)(df_embeddings['tweet'])


    # Then, expand contractions
    df_embeddings['tweet'] = preprocess.expand_acronyms (df_embeddings['tweet'], preprocess.english_contractions)
    
    
    # If the problem is classification, then we encode the label as numbers instead of using names
    if task_type == 'classification':
        
        # Create a binarizer
        lb = sklearn.preprocessing.LabelBinarizer ()
        
        
        # Get unique classes
        lb.fit (df_embeddings.label.unique ())
        
    
        # Note that we are dealing with binary (one label) or multi-class (one-hot enconding)
        if len (lb.classes_) > 2:
            df_labels = pd.DataFrame (lb.transform (df_embeddings["label"]), columns = lb.classes_)
            df_embeddings = pd.concat ([df_embeddings, df_labels], axis = 1)
            df_lf = pd.concat ([df_lf, df_labels], axis = 1)
            
        else:
            df_embeddings["label"] = df_embeddings["label"].astype ('category').cat.codes
            df_lf["label"] = df_lf["label"].astype ('category').cat.codes
    
    
    
    # @var number_of_classes int
    if task_type == 'regression':
        number_of_classes = 1
    
    elif len (lb.classes_) == 2:
        number_of_classes = 1
    
    else:
        number_of_classes = len (lb.classes_)
    
    
    # Note that we are keeping all the training and evaluation for the 
    # final model
    train_df_embeddings = df_embeddings
    train_df_lf = df_lf
    
    
    # @var Tokenizer Retrieve the tokenizer from disk
    token_filename = os.path.join (config.directories['assets'], args.dataset, key, 'tokenizer.pickle')
    with open (token_filename, 'rb') as handle:
        tokenizer = pickle.load (handle)
    

    # Update to tokens
    for dataframe in [train_df_embeddings]:
        dataframe['tokens'] = tokenizer.texts_to_sequences (dataframe['tweet'])
    
    
    # Get the max-len size
    maxlen = max (len (l) for l in train_df_embeddings['tokens'])
    
    
    # Transform sentences to tokens
    tokens = []
    for dataframe in [train_df_embeddings]:
        tokens.append (keras.preprocessing.sequence.pad_sequences (dataframe['tokens'], padding = 'pre', maxlen = maxlen))


    # Get the optimizers
    optimizers = [keras.optimizers.Adam]
    if task_type == 'regression':
        optimizers.append (keras.optimizers.RMSprop)
    
    
    # Get the reduction_metric
    reduction_metric = 'val_loss' if task_type == 'classification' else 'val_rmse'
    
    
    # Get best params
    params_filename = os.path.join (config.directories['assets'], args.dataset, key, 'hyperparameters-task-' + args.task + '.csv')
    df_params = pd.read_csv (params_filename, header = 0)
    df_params = df_params.sort_values (by = 'val_accuracy' if task_type == 'classification' else 'val_rmse', ascending = task_type == 'regression')
    
    
    # Get best values
    best_values = df_params.iloc[0]
    
    
    # Parameter space
    parameters_to_evaluate = {
        'task_type': [task_type],
        'tokenizer': [tokenizer],
        'name': [key],
        'dataset': [args.dataset],
        'number_of_classes': [number_of_classes],
        'epochs': [best_values['round_epochs']],
        'lr': [best_values['lr']],
        'optimizer': optimizers,
        'trainable': [True],
        'number_of_layers': [int (best_values['number_of_layers'])],
        'first_neuron': [int (best_values['first_neuron'])],
        'shape': [best_values['shape']],
        'batch_size': [best_values['batch_size']],
        'dropout': [float (best_values['dropout']) if best_values['dropout'] != 'False' else False],
        'kernel_size': [int (best_values['kernel_size'])],
        'maxlen': [int (maxlen)],
        'we_architecture': [best_values['we_architecture']],
        'activation': [best_values['activation']],
        'pretrained_embeddings': [best_values['pretrained_embeddings']],
        'features': [best_values['features']],
        'patience': [10]
    }
    
    
    # Get labels
    if task_type == 'classification' and len (lb.classes_) > 2:
        y = tensorflow.convert_to_tensor (train_df_embeddings[lb.classes_])
    
    else:
        y = tensorflow.convert_to_tensor (train_df_embeddings['label'])
    
    
    # Get features
    if task_type == 'classification':
        x = [tokens[0], train_df_lf.loc[:, ~train_df_lf.columns.isin ([*['label'], *lb.classes_])]]
    
    else:
        x = [tokens[0], train_df_lf.loc[:, ~train_df_lf.columns.isin ([*['label']])]]

    
    
    # @var time_limit
    time_limit = (datetime.datetime.now () + datetime.timedelta (minutes = args.minutes)).strftime ("%Y-%m-%d %H:%M")
    
    
    # and run the experiment saving weights
    scan_object = talos.Scan (
        x = x,
        y = y,
        x_val = x,
        y_val = y,
        params = parameters_to_evaluate,
        model = kerasModel.create,
        experiment_name = args.dataset,
        time_limit = time_limit,
        reduction_metric = reduction_metric,
        minimize_loss = True,
        print_params = True,
        round_limit = 1,
        save_weights = True,
        seed = config.seed
    )
    
    
    # Store
    model_filename = os.path.join (config.directories['assets'], args.dataset, key, 'model-task-' + args.task + '.h5')
    os.makedirs (os.path.dirname (model_filename), exist_ok = True)
    best_model = scan_object.best_model (reduction_metric, asc = True)
    
    best_model.save (model_filename)
    
    