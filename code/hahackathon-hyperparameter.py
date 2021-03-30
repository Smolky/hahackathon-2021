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
parser = argparse.ArgumentParser (description = 'Hyper-parameter optimization with TALOS')
parser.add_argument ('--dataset', dest = 'dataset', default = "hahackathon", help="|".join (config.datasets.keys ()))
parser.add_argument ('--force', dest = 'force', default = False, help="If True, it forces to replace existing files")
parser.add_argument ('--minutes', dest = 'minutes', default = 60 * 24 * 7, type = int, help = "Number of limits for the evaluation. Default is one week")
parser.add_argument ('--permutations', dest = 'permutations', default = None, type = int, help = "Max number of permutations")
parser.add_argument ('--patience', dest = 'patience', default = 10, type = int, help = "Patience for early stopping")
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
    
    
    # Divide the training dataset into training and validation with stratification
    # (in clase of classification)
    train_df_embeddings, val_df_embeddings = sklearn.model_selection.train_test_split (df_embeddings, train_size = dataset_options['train_size'], random_state = config.seed, stratify = df_embeddings['label'] if task_type == 'classification' else None)
    train_df_lf, val_df_lf = sklearn.model_selection.train_test_split (df_lf, train_size = dataset_options['train_size'], random_state = config.seed, stratify = df_lf['label'] if task_type == 'classification' else None)
    

    # @var Tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer (num_words = max_words, oov_token = True)
    
    
    # Fit on trainin dataset
    # @todo Preprocess
    tokenizer.fit_on_texts (pd.concat ([train_df_embeddings['tweet'], val_df_embeddings['tweet']]))

    
    # Store tokenizer
    token_filename = os.path.join (config.directories['assets'], args.dataset, key, 'tokenizer.pickle')
    os.makedirs (os.path.dirname (token_filename), exist_ok = True)
    with open (token_filename, 'wb') as handle:
        pickle.dump (tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)


    # Update to tokens
    for dataframe in [train_df_embeddings, val_df_embeddings]:
        dataframe['tokens'] = tokenizer.texts_to_sequences (dataframe['tweet'])
    
    
    # Get the max-len size
    maxlen = max (len (l) for l in train_df_embeddings['tokens'])

    
    # Transform sentences to tokens
    tokens = []
    for dataframe in [train_df_embeddings, val_df_embeddings]:
        tokens.append (keras.preprocessing.sequence.pad_sequences (dataframe['tokens'], padding = 'pre', maxlen = maxlen))
    
    
    # Get the optimizers
    optimizers = [keras.optimizers.Adam]
    if task_type == 'regression':
        optimizers.append (keras.optimizers.RMSprop)
    
    
    # Get the reduction_metric
    reduction_metric = 'val_loss' if task_type == 'classification' else 'val_rmse'
    
    
    # Parameter space
    parameters_to_evaluate = {
        'task_type': [task_type],
        'tokenizer': [tokenizer],
        'name': [key],
        'dataset': [args.dataset],
        'number_of_classes': [number_of_classes],
        'epochs': [1000],
        'lr': (0.5, 2, 10),
        'optimizer': optimizers,
        'trainable': [True],
        'number_of_layers': [1, 2, 3, 4, 5, 6, 7, 8],
        'first_neuron': [8, 16, 48, 64, 128, 256],
        'shape': ['funnel', 'rhombus', 'long_funnel', 'brick', 'diamond', 'triangle'],
        'batch_size': [16, 32, 64],
        'dropout': [False, 0.2, 0.5, 0.8],
        'kernel_size': [3, 5, 7],
        'maxlen': [maxlen],
        'we_architecture': ['dense', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru'],
        'activation': ['relu', 'sigmoid', 'tanh', 'selu', 'elu'],
        'pretrained_embeddings': ['none', 'fastText', 'glove'],
        'features': ['lf', 'we', 'lf+we'],
        'patience': [args.patience]
    }
    
    
    # Get labels
    if task_type == 'classification' and len (lb.classes_) > 2:
        y = tensorflow.convert_to_tensor (train_df_embeddings[lb.classes_])
        y_val = tensorflow.convert_to_tensor (val_df_embeddings[lb.classes_])
    
    else:
        y = tensorflow.convert_to_tensor (train_df_embeddings['label'])
        y_val = tensorflow.convert_to_tensor (val_df_embeddings['label'])
        
    
    # Get features
    if task_type == 'classification':
        x = [tokens[0], train_df_lf.loc[:, ~train_df_lf.columns.isin ([*['label'], *lb.classes_])]]
        x_val = [tokens[1], val_df_lf.loc[:, ~val_df_lf.columns.isin ([*['label'], *lb.classes_])]]
    
    else:
        x = [tokens[0], train_df_lf.loc[:, ~train_df_lf.columns.isin ([*['label']])]]
        x_val = [tokens[1], val_df_lf.loc[:, ~val_df_lf.columns.isin ([*['label']])]]
    
    
    # @var time_limit
    time_limit = (datetime.datetime.now () + datetime.timedelta (minutes = args.minutes)).strftime ("%Y-%m-%d %H:%M")
    
    
    # and run the experiment
    scan_object = talos.Scan (
        x = x,
        x_val = x_val,
        y = y,
        y_val = y_val,
        params = parameters_to_evaluate,
        model = kerasModel.create,
        experiment_name = args.dataset,
        time_limit = time_limit,
        reduction_metric = reduction_metric,
        minimize_loss = True,
        print_params = True,
        round_limit = args.permutations,
        save_weights = False,
        seed = config.seed
    )
    
    
    # Store scan results for further analysis
    # We remove some values that do not changed
    """    
    columns_to_drop = []
    
    for column_name, value in parameters_to_evaluate.items ():
        if len (value) == 1:
            columns_to_drop.append (column_name)
    
    scan_object.data = scan_object.data.drop (columns_to_drop, axis = 1)
    """
    
    # One hot encoding
    for feature in ['shape', 'we_architecture', 'activation', 'pretrained_embeddings']:
        if feature in scan_object.data.columns:
            scan_object.data = utils.pd_onehot (scan_object.data, feature)
    
    
    # Store
    params_filename = os.path.join (config.directories['assets'], args.dataset, key, 'hyperparameters-task-' + args.task + '.csv')
    os.makedirs (os.path.dirname (params_filename), exist_ok = True)
    scan_object.data.to_csv (params_filename, index = False)