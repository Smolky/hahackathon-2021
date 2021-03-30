'''
    Utils
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
'''

import sys
import math
import netrc
import requests
import os
import random
import config
import numpy as np
import pandas as pd
import pkbar

from requests.auth import AuthBase
from pathlib import Path

def pd_onehot (df, feature):
    """
    pd_onehot
    
    @link https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    """
    dummies = pd.get_dummies (df[[feature]])
    res = pd.concat([df, dummies], axis=1)
    return (res)
    
    
    
def get_neurons_per_layer (shape, number_of_layers, first_neuron):
    """
    Return the number of neurons per layer for MLP according to a shape
    
    @param shape
    @param number_of_layers
    @param first_neuron
    """
    
    # @var neurons_per_layer List Contains a list of the neurons per layer. Used for the shape
    neurons_per_layer = []    
    
    # Calculate the number of neurons per layer
    # In 'funnel' shape, the number of layers is decreasing
    if (shape == 'funnel'):
        for i in range (1, number_of_layers + 1):
            neurons_per_layer.append (round (first_neuron / i))
        
    
    # In 'rhombus' the first layer equals to 1 and the next layers slightly increase until the middle one 
    # which equals to neuron size
    if (shape == 'rhombus'):
        
        # Init the neurons to 1
        neurons_per_layer = [1] * number_of_layers
        
        
        # Iterate only in the first half of the array
        for first_index in range (round (number_of_layers / 2)):
            
            # @var half int The central point of the array
            half = math.ceil (number_of_layers / 2) - 1
            
            
            # @var distance int The distance between the half and the first index
            distance = abs (first_index - half)
            
            
            # @var mirror_index int The opposite index
            mirror_index = half + distance
            
            
            # Update opposite index with the same value
            neurons_per_layer[first_index] = int (first_neuron / pow (2, distance))
            neurons_per_layer[mirror_index] = neurons_per_layer[first_index]


    # In 'long_funnel' the first half of the layers have the value of neuron_max
    if (shape == 'long_funnel'):
    
        # Init the neurons to 1
        neurons_per_layer = [1] * number_of_layers
        
        
        # Iterate over all the layers
        for index in range (number_of_layers):
            
            # Get half position
            half = math.ceil (number_of_layers / 2) - 1
            
            
            # Set value
            neurons_per_layer[index] = round (first_neuron / (pow (2, index - half))) if index > half else 1
    
    
    # In 'brick' all neurons have the same number of neurons
    if (shape == 'brick'):
        neurons_per_layer = [first_neuron] * number_of_layers
    
    
    # In 'diamond' the shape is similar to rhombus but with open first half
    if (shape == 'diamond'):
        
        # Init the neurons to 1
        neurons_per_layer = [1] * number_of_layers
        
        
        # Iterate over the half of layers
        for first_index in range (round (number_of_layers / 2)):
            
            # @var half int The central point of the array
            half = math.ceil (number_of_layers / 2) - 1
            
            
            # Get the distance
            distance = abs (first_index - half)
            
            
            # @var mirror_index int The opposite index
            mirror_index = half + distance
            
            
            # Update opposite index with the same value
            neurons_per_layer[first_index] = int (first_neuron / 2)
            neurons_per_layer[mirror_index] = int (first_neuron / pow (2, distance))

    
    # In 'triangle' the shape is similar to a funnel but reversed
    if (shape == 'triangle'):
    
        for i in range (1, number_of_layers + 1):
            neurons_per_layer.append (round (first_neuron / i))    
        neurons_per_layer = neurons_per_layer[::-1]
        
        
    return neurons_per_layer        
    
    
"""
   get_embedding_matrix
   
   @param key String The key of the pretained word embedding
   @param tokenizer Tokenizer
   @param experiment String
   @param dataset String
   @param embedding_dim int
   @param force Boolean
   
   @link https://realpython.com/python-keras-text-classification/#your-first-keras-model
"""

def get_embedding_matrix (key, tokenizer, experiment, dataset, embedding_dim = 300, force = False):
    
    # Cache file
    cache_file = os.path.join (config.directories['cache'], experiment, dataset, key + '.npy')
    
    
    # Restore cache
    if (not force and os.path.isfile (cache_file)):
        return np.load (cache_file)
    
    
    # Adding again 1 because of reserved 0 index
    vocab_size = len (tokenizer.word_index) + 1  
    
    
    # embedding_matrix
    embedding_matrix = np.zeros ((vocab_size, embedding_dim))


    # Get num lines for the progress bar
    num_lines = sum (1 for line in open (config.pretrained_models[key]['vectors']))


    # Create pkbar
    pbar = pkbar.Pbar (name = 'generating ' + key, target = num_lines)
    
    
    # Open vector file
    index = 0
    with open (config.pretrained_models[key]['vectors'], 'r', encoding = 'utf8', errors = 'ignore') as f:
        
        for line in f:
            
            # Get word and vector
            word, *vector = line.split ()
            
            
            # Check if the word is in the word index
            if word in tokenizer.word_index:
            
                # Get position
                idx = tokenizer.word_index[word] 
                
                
                # Set weights
                embedding_matrix[idx] = np.array (vector, dtype = np.float32)[:embedding_dim]
                
            
            # Update list
            index += 1
            pbar.update (index)
                

    # Store in cache
    os.makedirs (os.path.dirname (cache_file), exist_ok = True)
    np.save (cache_file, embedding_matrix)


    # Return embedding matrix
    return embedding_matrix