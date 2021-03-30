"""
    Obtain the datasets and place them at the dataset folders
    
    To run, this code needs environment variables for the authentication
    of UMUCorpusClassifier platform.

    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import requests
import sys
import csv
import itertools
import netrc
import os.path
import config
import argparse
import io
import pkbar
import pandas as pd
import math
import string

import fasttext

from pathlib import Path


class Dataset ():
    """
    Dataset
    """

    # The dataframe
    df = None

    
    def __init__ (self, dataset, options, refresh = False):
        """
        Constructor
        
        @param dataset
        @param options
        @param refresh
        """
        self.dataset = dataset
        self.options = options
        self.refresh = refresh
    
    
    def set (self, df):
        """
        @param
        """
        self.df = df
    
    
    def get (self):
        """
        Retrieve the Pandas Dataframe
        
        @return dataframe
        """
        
        if self.df is None:
            
            # filename
            filename = os.path.join (config.directories['datasets'], self.dataset)
            
            
            # Get the dataset as a dataframe
            if not self.refresh and os.path.isfile (filename):
                self.df = pd.read_csv (os.path.join (config.directories['datasets'], self.dataset), header = 0, sep = ",")

            
            else:
                self.df = self.compile ()
        
        
        # Update dataframe types
        for column in ['tweet', 'tagged']:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype (str)
        
        
        return self.df
    
    
    def calculate_language (self, label = "tweet", default_language = 'es', threshold = 0.75):
        """
        @param label
        @param default_language
        @param threshold
        """
        
        # @var model This model is used to get the lang
        model = fasttext.load_model (os.path.join (config.directories['pretrained'], 'lid.176.bin'))
        
        
        # Get the tweets without punctuation signs 
        tweet = self.df[label].str.translate (str.maketrans ('', '', string.punctuation))
        
        # 234661 --- 241863
        
        
        def func (tweet):
            """
            """
            
            # prediction
            prediction = model.predict (tweet, k = 2)
            
            
            # @var first_language String
            first_language = prediction[0][0].replace ('__label__', '')


            # @var first_language_affinity int
            first_language_affinity = prediction[1][0]
            
            
            # Return the language if exceedes the threshold
            return first_language if first_language_affinity > threshold else None
        
        
        # Assign the language
        self.df['language'] = self.df['tweet'].apply (func)

        
        return self.df
    
    
    def compile (self):
        """ 
        Compiles the dataset 
        """

        # @var auth_token String
        auth_token = config.get_auth_token (config.umucorpusclassifier_api_endpoint, config.certificate)
        
        
        # @request_payload Dir
        request_payload = {
            'export-format': 'csv',
            'size': self.options['max'],
            'corpora[]': ','.join (str(x) for x in self.options['ids']),
            'preprocessing[]': self.options['preprocessing'],
            'fields[]': self.options['fields']
        }
        
        
        # Attach strategy (if specified)
        if ('strategy' in self.options):
            request_payload['strategy'] = self.options['strategy'];
        
        
        # Attach if the corpus is balanced
        if ('balanced' in self.options):
            request_payload['balanced'] = True;

        if ('filter-date-start' in self.options):
            request_payload['filter-date-start'] = self.options['filter-date-start'];

        if ('filter-date-end' in self.options):
            request_payload['filter-date-end'] = self.options['filter-date-end'];
        
        
        
        # @var reponse Response
        response = requests.post (
            config.umucorpusclassifier_api_endpoint + 'admin/export-corpus.csv', 
            json = request_payload, 
            verify = config.certificate,
            auth = config.PLNAuth (auth_token),
        )
        
        
        # Check response
        if (response.status_code == 401):
            print ("Authentication failed: " + str (response.status_code))
            print (response.text)
            print (request_payload)
            sys.exit ()
            
        if (response.status_code != 200):
            print ("Request failed: " + str (response.status_code))
            print (response.text)
            print (request_payload)
            sys.exit ()
        
        
        # Store compiled dataset to disk
        s = response.content
        return pd.read_csv (io.StringIO (s.decode ('utf-8')))

    

def main ():
    """ To use from command line """
    
    # Parser
    parser = argparse.ArgumentParser (description = 'Generates the dataset')
    parser.add_argument ('--dataset', dest = 'dataset', default=next (iter (config.datasets)), help="|".join (config.datasets.keys ()))
    parser.add_argument ('--force', dest = 'force', default = False, help = "If True, it forces to replace existing files")


    # Get args
    args = parser.parse_args ()
    
    
    # datasets
    datasets = config.datasets[args.dataset].items ()
    
    
    # @var umucorpus_ids int|string The Corpus IDs
    for key, dataset_options in datasets:
        
        # dataset_name
        dataset_name = args.dataset + "-" + key + ".csv"
        
    
        # Get dataset
        dataset = Dataset (dataset_name, dataset_options, args.force)
    
    
        # Retrieve the dataset
        df = dataset.get ()
        
        
        # Save to disk
        df.to_csv (os.path.join (config.directories['datasets'], dataset_name), index = False, quoting = csv.QUOTE_ALL)


if __name__ == "__main__":
    main ()
