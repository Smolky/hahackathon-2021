"""
    Hahackathon BERT
    
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
epochs = 10


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
    
    
    # @var String The name of the 
    model_filename = os.path.join (config.directories['assets'], args.dataset, key, 'bert-finetunning-' + args.task)
    
    
    # Retrieve the final fine tuned model
    pretrained_model = model_filename
    
    
    # Get the model
    class CustomBERTModel (nn.Module):
        """
        CustomBERTModel
        
        This model mixes the fine tunned BERT model with custom features based 
        on linguistic features
        """
        
        def __init__ (self, input_size, num_classes):
            """
            @param input_size
            @param num_classes
            """
            super (CustomBERTModel, self).__init__()
            
            
            # Init BERT model
            self.bert = BertForSequenceClassification.from_pretrained (pretrained_model, return_dict = True, output_hidden_states = True)
            
            
            # Linguistic features layer
            self.fc1 = nn.Linear (input_size + (768 * 1), 32)
            self.fc2 = nn.Linear (32, 16)
            self.fc3 = nn.Linear (16, num_classes)
        
        
        def forward (self, lf, input_ids, token_type_ids = None, attention_mask = None, position_ids = None, head_mask = None, epoch = 0):
            
            # Get BERT results
            with torch.no_grad ():
                sequence_output = self.bert (input_ids, attention_mask = attention_mask)
            
            
            # Get BERT hidden_states
            hidden_states = sequence_output.hidden_states
            
            
            # @var cls_tokens The first token for each batch
            # @link https://stackoverflow.com/questions/61465103/how-to-get-intermediate-layers-output-of-pre-trained-bert-model-in-huggingface
            
            # This way works fine, getting the last layer
            cls_tokens = hidden_states[-1][:, 0]
            
            
            # To test
            """
            token_embeddings = hidden_states[-2][:, 1:]
            token_embeddings = torch.mean (token_embeddings, dim = -2)
            """
            
            
            # Combine BERT with LF
            combined_with_bert_features = torch.cat ((cls_tokens, lf), dim = 1)
            
            
            # Handle LF
            lf_x = F.relu (self.fc1 (combined_with_bert_features))
            lf_x = F.relu (self.fc2 (lf_x))
            
            
            # According to the task type, we need to apply a sigmoid function
            # or return the value as it is
            if task_type == 'classification':
                lf_x = torch.sigmoid (self.fc3 (lf_x))
            else:
                lf_x = self.fc3 (lf_x)
            
            return lf_x

    
    # Get the dataset as a dataframe
    BERT_df = dataset.get ()
    BERT_df = BERT_df.drop (BERT_df[BERT_df['is_test'] == True].index)
    BERT_df = dataset.getDFFromTask (args.task, BERT_df)
    
    
    # Determine the task type
    task_type = 'classification' if args.task in ['1a', '1c'] else 'regression'

    
    # Preprocess. First, some basic stuff
    for pipe in ['remove_urls', 'remove_digits', 'remove_whitespaces', 'remove_elongations', 'to_lower', 'remove_punctuation']:
        BERT_df['tweet'] = getattr (preprocess, pipe)(BERT_df['tweet'])


    # Then, expand contractions
    BERT_df['tweet'] = preprocess.expand_acronyms (BERT_df['tweet'], preprocess.english_contractions)
    
    
    
    # Get linguistic features
    df_lf = pd.read_csv (os.path.join (config.directories['assets'], args.dataset, key, 'lf.csv'), header = 0, sep = ",")
    df_lf = df_lf.rename (columns = {"class": "label"})
    df_lf = df_lf.loc[:, (df_lf != 0).any (axis = 0)]
    df_lf = df_lf.loc[BERT_df.index]
    df_lf_x = df_lf.iloc[:, :-1]
    
    
    # Merge both dataframes and push the label column to the last position
    df = pd.concat ([BERT_df, df_lf_x], axis=1)
    df = df[[c for c in df if c not in ['label']] + ['label']]
    
    
    # Test
    # Very important to move
    if (task_type == 'regression'):
        df = df.drop (df[df['label'] == 0.0].index).reset_index ()
        df_lf_x = df_lf_x.loc[df.index]
    
    
    # The input size is the number of linguistic features. We get this value from the 
    # dataframe, but we remove 2 items. One for the "TWEET" column, and other for the 
    # class    
    input_size = df_lf_x.shape[1]
    
    
    # CustomBERTModel model
    model = CustomBERTModel (input_size, num_classes = len (df['label'].unique ()) if task_type == 'classification' else 1)
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
        torch.tensor (df_lf_x.values), 
        dataset['input_ids'],
        dataset['attention_mask'],
        dataset['label']
    )
    
    
    # Split the dataframes into training, validation and testing
    if args.evaluate:
        
        # Split the dataframes into training, validation and testing
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
    
        train_loader = torch.utils.data.DataLoader (dataset, batch_size = batch_size, shuffle = False)

    
    # Set the AdamW optimizer
    optimizer = AdamW (model.parameters (), lr = 1e-2)
    
    
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
        
        
        # Store our loss and accuracy for plotting
        train_loss_set = []
        
        
        # Store correct predictions
        correct_predictions = 0
        
        
        # Train this epoch
        model.train ()
        
        
        # Get all batches in this epoch
        for i, (lf, input_ids, attention_mask, labels) in enumerate (train_loader):
        
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates 
            # the gradients on subsequent backward passes. This is convenient while training RNNs. 
            # So, the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            # @link https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad ()
            
            
            # Move features to device
            input_ids = input_ids.to (device)
            attention_mask = attention_mask.to (device)
            labels = labels.to (device)
            lf = lf.float ().to (device)
            
            
            # Forward model
            predictions = model (lf, input_ids, attention_mask = attention_mask)
            
            
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
            
            
            # Do deep learning stuff
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
            model.eval ()
            
            
            # Store correct predictions
            correct_predictions = 0
            
            
            # Store loss
            val_losses = []
            
            
            # No gradient is needed
            with torch.no_grad ():
                for i, (lf, input_ids, attention_mask, labels) in enumerate (test_loader):
                
                    # Move features to device
                    input_ids = input_ids.to (device)
                    attention_mask = attention_mask.to (device)
                    labels = labels.to (device)
                    lf = lf.float ().to (device)
                
                
                    # Forward model
                    predictions = model (lf, input_ids, attention_mask = attention_mask)
                    
                    
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
                    
                    
                    # Get BCE loss for each batch
                    val_losses.append (loss.item ())
                
                
                # Update values
                kbar_values = [("val_loss", np.mean (val_losses))]
                if task_type == 'classification':
                    kbar_values.append (('val_acc', correct_predictions.item () / val_df.shape[0]))
            
            
                # Update var
                kbar.add (0, values = kbar_values)


    # Store
    if not args.evaluate:
    
        model_filename = os.path.join (config.directories['assets'], args.dataset, key, 'bert-final-' + args.task + '.pt')
        os.makedirs (os.path.dirname (model_filename), exist_ok = True)
        
        
        # Save model
        torch.save (model.state_dict (), model_filename)
