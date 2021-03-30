"""
    Hahackathon BERT fine-tunning
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
    
    Useful links
    @link https://colab.research.google.com/drive/1-JIJlao4dI-Ilww_NnTc0rxtp-ymgDgM?usp=sharing#scrollTo=utjMLdmqsUuA
"""

# Import libraries
import sys
import math
import csv
import torch
import argparse
import pandas as pd
import numpy as np
import pkbar
import config
import os
import torch.nn.functional as F
import torch.utils.data as data_utils

from transformers import BertForSequenceClassification, BertModel, BertTokenizerFast, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn

from datasetResolver import DatasetResolver
from preprocessText import PreProcessText


# Define batch size
batch_size = 64


# Define the number of epochs
epochs = 2


# Get device
device = torch.device ('cuda') if torch.cuda.is_available () else torch.device ('cpu')


# Config seed for reproductibility
torch.manual_seed (config.seed)


def tokenize (batch):
    """
    Bert Tokenizer
    """
    return tokenizer (batch['tweet'], padding = True, truncation = True)



# Parser
parser = argparse.ArgumentParser (description='Generate hahackaton-bert')
parser.add_argument ('--dataset', dest='dataset', default="hahackathon", help="|".join (config.datasets.keys ()))
parser.add_argument ('--label', dest='label', default="label", help="The label to classify")
parser.add_argument ('--force', dest='force', default=False, help="If True, it forces to replace existing files")
parser.add_argument ('--task', dest = 'task', default = "1a", help = "Get the task")
parser.add_argument ('--evaluate', dest = 'evaluate', type = lambda x: (str(x).lower() in ['True', 'true','1', 'yes']), default = True, help = "Evaluate with test")


# Get args
args = parser.parse_args ()


# Set the pretrained model 
pretrained_model = 'bert-base-uncased'


# Get the model
class CustomBERTModelFineTunning (nn.Module):

    def __init__ (self, num_labels = None):
        super (CustomBERTModelFineTunning, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained (pretrained_model, return_dict = True, num_labels = num_labels)
        self.bert.to (device)
        
        
    def forward (self, input_ids, token_type_ids = None, attention_mask = None, position_ids = None, head_mask = None):
        bert_x = self.bert (input_ids, attention_mask = attention_mask)
        return bert_x.logits


# Preprocess text
preprocess = PreProcessText ()


# test
models = []


# @var umucorpus_ids int|string The Corpus IDs
for key, dataset_options in config.datasets[args.dataset].items ():
    
    # Resolver
    resolver = DatasetResolver ()


    # Get the dataset name
    dataset_name = args.dataset + "-" + key + '.csv'


    # Get the dataset
    dataset = resolver.get (dataset_name, dataset_options, args.force)


    # Get the dataset as a dataframe
    df = dataset.get ()
    df = df.drop (df[df['is_test'] == True].index)
    df = dataset.getDFFromTask (args.task, df)
    
    
    # Determine the task type
    task_type = 'classification' if args.task in ['1a', '1c'] else 'regression'
    
    
    # Test
    # Very important to move
    if (task_type == 'regression'):
        df = df.drop (df[df['label'] == 0.0].index).reset_index ()
    
    
    # Preprocess. First, some basic stuff
    for pipe in ['remove_urls', 'remove_digits', 'remove_whitespaces', 'remove_elongations', 'to_lower', 'remove_punctuation']:
        df['tweet'] = getattr (preprocess, pipe)(df['tweet'])
        
        
    # Then, expand contractions
    df['tweet'] = preprocess.expand_acronyms (df['tweet'], preprocess.english_contractions)
    
    
    
    # CustomBERTModel model
    model = CustomBERTModelFineTunning (num_labels = len (df['label'].unique ()) if task_type == 'classification' else 1)
    model.to (device)
    
    
    # Get the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained (pretrained_model)
    
    
    # Encode label as numbers instead of user names
    if task_type == 'classification':
        df["label"] = df["label"].astype ('category').cat.codes
    
    
    # Encode datasets to work with transformers
    dataset = Dataset.from_pandas (df)
    
    
    # Tokenizer trainset and test dataframe with the training
    # The tokenize function only takes care of the "tweet"
    # column and will create the input_ids, token_type_ids, and 
    # attention_mask
    dataset = dataset.map (tokenize, batched = True, batch_size = len (dataset))
    
    
    # Finally, we "torch" the new columns. We return the rest 
    # of the columns with "output_all_columns"
    dataset.set_format ('torch', columns = ['input_ids', 'attention_mask', 'label'], output_all_columns = True)

    
    # Create a dataset with the linguistic features joined, the input id, the attention mask, and the labels
    dataset = data_utils.TensorDataset (
        dataset['input_ids'],
        dataset['attention_mask'],
        dataset['label']
    )
    
    
    # Split the dataframes into training, validation and testing
    if args.evaluate:
        
        # We split our dataset and we stratify it in classification problems
        train_df, val_df = train_test_split (df, 
            train_size = dataset_options['train_size'], 
            random_state = config.seed, 
            stratify = df[['label']] if task_type == 'classification' else None
        )
        
        
        # Generate a sampler from the indexes
        train_sampler = torch.utils.data.SubsetRandomSampler (train_df.index)
        val_sampler = torch.utils.data.SubsetRandomSampler (val_df.index)
        
        
        # Create the loader
        train_loader = torch.utils.data.DataLoader (dataset, batch_size = batch_size, sampler = train_sampler, shuffle = False)
        test_loader = torch.utils.data.DataLoader (dataset, batch_size = batch_size, sampler = val_sampler, shuffle = False)

    else:

        # Get all data for training
        train_loader = torch.utils.data.DataLoader (dataset, batch_size = batch_size, shuffle = False)

    
    # Set the AdamW optimizer
    # @todo Several values
    optimizer = AdamW (model.parameters (), lr = 1e-4)
    
    
    # Set the loss Criteria according to the task type
    criterion = torch.nn.CrossEntropyLoss () if task_type == 'classification' else torch.nn.MSELoss ()
    criterion = criterion.to (device)
    
    
    # Define the size of the progress bar
    if args.evaluate:
        train_per_epoch = len (train_df) / batch_size
        eval_per_epoch = len (val_df) / batch_size
    else:
        train_per_epoch = len (df) / batch_size
    
    
    
    # Get metrics
    if args.evaluate:
        metrics = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc'] if task_type == 'classification' else ['epoch', 'loss', 'val_loss']
        
    else:
        metrics = ['epoch', 'loss', 'acc'] if task_type == 'classification' else ['epoch', 'loss']
        
    
    # Train and eval each epoch
    for epoch in range (1, epochs + 1):
        
        # Create a progress bar
        # @link https://github.com/yueyericardo/pkbar/blob/master/pkbar/pkbar.py (stateful metrics)
        kbar = pkbar.Kbar (target = train_per_epoch, width = 32, stateful_metrics = metrics)
        
        
        # Store our loss
        train_loss_set = []
        
        
        # Store correct predictions
        correct_predictions = 0
        
        
        # Train this epoch
        model.train ()
        
        
        # Get all batches in this epoch
        for i, (input_ids, attention_mask, labels) in enumerate (train_loader):
        
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates 
            # the gradients on subsequent backward passes. This is convenient while training RNNs. 
            # So, the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            # @link https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad ()
            
            
            # Move features to device
            input_ids = input_ids.to (device)
            attention_mask = attention_mask.to (device)
            labels = labels.to (device)
            
            
            # Forward model
            predictions = model (input_ids, attention_mask = attention_mask)
            
            
            # If the domain is classification, then get the correct predictions
            if task_type == 'classification':
                predictions = torch.squeeze (predictions)
                
                
                # Use max to get the correct class
                _, preds = torch.max (predictions, dim = 1)
                
                
                # Get the correct predictions
                correct_predictions += torch.sum (preds == labels)
                
                
                # Get the accuracy
                acc = correct_predictions.item ()  / (batch_size * (i + 1))
            
            
                # Get loss
                loss = criterion (predictions, labels)
            
            # If the problem is regression
            else:
            
                # Get loss
                loss = torch.sqrt (criterion (predictions, labels.float ()) + 1e-6)
            
            
            # Store loss
            train_loss_set.append (loss.item ())
            
            
            # Do deep-learning stuff...
            loss.backward ()
            optimizer.step ()
            
            
            # Update metrics in each step
            kbar_values = [('epoch', int (epoch)), ("loss", loss.item ())]
            if task_type == 'classification':
                kbar_values.append (('acc', acc))
            
            kbar.add (1, values = kbar_values)
            
        
        # Evaluate
        if args.evaluate:
            
            # Eval this epoch with the test
            model = model.eval ()
            
            
            # Store correct predictions
            correct_predictions = 0
            
            
            # Store loss
            val_losses = []
            
            
            # No gradient is needed
            with torch.no_grad ():
                for i, (input_ids, attention_mask, labels) in enumerate (test_loader):
                
                    # Move features to device
                    input_ids = input_ids.to (device)
                    attention_mask = attention_mask.to (device)
                    labels = labels.to (device)
                    
                    
                    # Forward model
                    predictions = model (input_ids, attention_mask = attention_mask)
                    
                    
                    # If the domain is classification, then get the correct predictions
                    if task_type == 'classification':
                        predictions = torch.squeeze (predictions)
                        
                        
                        # Use max to get the correct class
                        _, preds = torch.max (predictions, dim = 1)
                        
                        
                        # Get the correct predictions
                        correct_predictions += torch.sum (preds == labels)
                        
                        
                        # Get loss
                        loss = criterion (predictions, labels)
                    
                    # If the problem is regression
                    else:
                    
                        # Get loss
                        loss = torch.sqrt (criterion (predictions, labels.float ()) + 1e-6)
                    
                    
                    # Get BCE loss
                    val_losses.append (loss.item ())
                
                
                # Update values
                kbar_values = [("val_loss", np.mean (val_losses))]
                if task_type == 'classification':
                    kbar_values.append (('val_acc', correct_predictions.item () / val_df.shape[0]))
            
            
                # Update var
                kbar.add (0, values = kbar_values)


    # Store
    if not args.evaluate:
    
        model_filename = os.path.join (config.directories['assets'], args.dataset, key, 'bert-finetunning-' + args.task)
        os.makedirs (os.path.dirname (model_filename), exist_ok = True)
        

        # Save model
        model.bert.save_pretrained (model_filename)
        tokenizer.save_pretrained (model_filename)

