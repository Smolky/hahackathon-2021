# https://colab.research.google.com/drive/1-JIJlao4dI-Ilww_NnTc0rxtp-ymgDgM?usp=sharing#scrollTo=utjMLdmqsUuA
# 
# To test transfomers
import sys
import csv
import torch
import argparse
import pandas as pd
import numpy as np
import pkbar
import config
import os

from transformers import BertForSequenceClassification, BertTokenizerFast, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch import nn

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from datasetResolver import DatasetResolver


# Parser
parser = argparse.ArgumentParser (description='Generate hahackaton-bert')
parser.add_argument ('--dataset', dest='dataset', default=next (iter (config.datasets)), help="|".join (config.datasets.keys ()))
parser.add_argument ('--label', dest='label', default="label", help="The label to classify")
parser.add_argument ('--force', dest='force', default=False, help="If True, it forces to replace existing files")


# Train ratio (float or int, default=None)
train_size = 0.8


# Validation ratio. This value will be used to split the train split again, achieving training 60 testing 20 eval 20
train_val_size = 0.75


# Define batch size
batch_size = 64


# Define the number of epochs
epochs = 2


# Get device
device = torch.device ('cuda') if torch.cuda.is_available () else torch.device ('cpu')


# Config seed for reproductibility
torch.manual_seed (config.seed)
np.random.seed (config.seed)


def tokenize (batch):
    """
    Bert Tokenizer
    """
    return tokenizer (batch['tweet'], padding=True, truncation=True)



# Set the pretrained model 
pretrained_model = 'bert-base-uncased'


# Get the model
class BertForSequenceRegression (nn.Module):
    """
    Define a custom class for handling BERT in combination with the linguistic features
    """
    def __init__ (self):
        super (BertForSequenceRegression, self).__init__()
        
        # Bert layer
        self.bert = BertForSequenceClassification.from_pretrained (pretrained_model, num_labels = 1, return_dict = True)
        self.bert.to (device)
    
    def forward (self, input_ids, token_type_ids = None, attention_mask = None, position_ids = None, head_mask = None, epoch = 1):
        
        # Handle BERT
        bert_x = self.bert (input_ids, attention_mask = attention_mask)
        
        
        return bert_x.logits


# Get args
args = parser.parse_args ()



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
    
    df['tweet'] = df['tweet'].str.lower ()
    
    
    # Copy the label
    if args.label != 'label':
        df['label'] = df[args.label]
    
    
    
    # Get the model
    model = BertForSequenceRegression ()
    model.to (device)


    # Get the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained (pretrained_model)
    


    # Split the dataframes into training, validation and testing
    train_df, test_df = train_test_split (df, train_size = train_size, random_state = config.seed)
    train_df, val_df = train_test_split (train_df, train_size = train_val_size, random_state = config.seed)
    
    
    # Encode datasets to work with PyTorch
    train_dataset = Dataset.from_pandas (train_df)
    test_dataset = Dataset.from_pandas (test_df)
    val_dataset = Dataset.from_pandas (val_df)
    

    # Tokenizer trainset and test dataframe with the training
    train_dataset = train_dataset.map (tokenize, batched = True, batch_size = len (train_dataset))
    test_dataset = test_dataset.map (tokenize, batched = True, batch_size = len (test_dataset))
    val_dataset = test_dataset.map (tokenize, batched = True, batch_size = len (val_dataset))


    # Set torch format to specific columns to use with BERT
    train_dataset.set_format ('torch', columns = ['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format ('torch', columns = ['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format ('torch', columns = ['input_ids', 'attention_mask', 'label'])
    
    
    # Set the AdamW optimizer
    optimizer = AdamW (model.parameters (), lr = 1e-4)
    
    
    # Set the loss Criteria
    # https://stackoverflow.com/questions/61990363/rmse-loss-for-multi-output-regression-problem-in-pytorch
    criterion = torch.nn.MSELoss ()
    criterion.to (device)
    
    
    # Train loader
    train_loader = DataLoader (train_dataset, batch_size = batch_size, shuffle = False)


    # Test loader
    test_loader = DataLoader (test_dataset, batch_size = batch_size, shuffle = False)
    
    
    # Val loader
    val_loader = DataLoader (val_dataset, batch_size = batch_size, shuffle = False)
    
    
    # Define the size of the progress bar
    train_per_epoch = len (train_df) / batch_size
    
    
    # Train and eval each epoch
    for epoch in range (epochs):
        
        # Create a progress bar
        # @link https://github.com/yueyericardo/pkbar/blob/master/pkbar/pkbar.py (stateful metrics)
        kbar = pkbar.Kbar (target = train_per_epoch, width = 32, stateful_metrics = ['loss', 'acc', 'epoch', 'val_loss', 'val_acc'])
        
        
        # Store our loss and accuracy for plotting
        train_loss_set = []
        
        
        # Store correct predictions
        correct_predictions = 0
        i = 0
        
        
        # Train this epoch
        model.train ()
        
        
        # Get all batches in this epoch
        for batch in train_loader:
        
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates 
            # the gradients on subsequent backward passes. This is convenient while training RNNs. 
            # So, the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            # @link https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad ()
            
            
            # Move features to device
            input_ids = batch['input_ids'].to (device)
            attention_mask = batch['attention_mask'].to (device)
            labels = batch['label'].to (device)
            
            
            # Get predictions
            predictions = model (input_ids, attention_mask = attention_mask, epoch = epoch)
            
            
            # Get loss
            loss = torch.sqrt (criterion (predictions, labels))
            
            
            # Store loss
            train_loss_set.append (loss.item ())
            
            
            # Increment the iterator
            i += 1
            
            
            # Update the neural network
            loss.backward ()
            optimizer.step ()
            
            
            # Update the progress bar
            kbar.add (1, values=[("loss", loss.item ()), ('epoch', int (epoch))])
            
        
        
        # Eval this epoch with the test
        model = model.eval ()
        
        
        # Store correct predictions
        correct_predictions = 0
        
        
        # Store loss
        val_losses = []
        
        
        # No gradient is needed
        with torch.no_grad ():
            for batch in test_loader:
            
                # Move features to device
                input_ids = batch['input_ids'].to (device)
                attention_mask = batch['attention_mask'].to (device)
                labels = batch['label'].to (device)
                
                
                # Get predictions
                predictions = model (input_ids, attention_mask = attention_mask, epoch = epoch)
                
                
                # Get loss
                loss = torch.sqrt (criterion (predictions, labels))
                
                
                # Get BCE loss
                val_losses.append (loss.item ())

        
        # Update var
        kbar.add (0, values=[("val_loss", np.mean (val_losses))])


