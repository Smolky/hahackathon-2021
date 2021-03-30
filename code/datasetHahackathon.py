"""
    Hahackathon dataset
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import config
import fasttext
import config
import re
import regex
import argparse
import csv
import sys
import pkbar
import math
import string
import numpy as np
import pandas as pd
import os

from dataset import Dataset
from scipy import stats
from preprocessText import PreProcessText



class DatasetHahackathon (Dataset):
    """
    DatasetHahackathon
    
    @extends Dataset
    """

    def __init__ (self, dataset, options, refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, refresh)


    def compile (self):
        """
        This file is already exists
        """
        
        # Load dataframes
        df_train = pd.read_csv (os.path.join (config.directories['datasets'], "hahackathon_train.csv"))
        df_test = pd.read_csv (os.path.join (config.directories['datasets'], "hahackathon_test.csv"))
        
        
        # Create columns to distinguish between train and test
        df_train = df_train.assign (is_train = True)
        df_train = df_train.assign (is_test = False)
        df_test = df_test.assign (is_train = False)
        df_test = df_test.assign (is_test = True)
        
        
        # Merge
        df = pd.concat ([df_train, df_test], axis = 0, ignore_index = True)
        
        
        # Change class names
        df = df.rename (columns = {'text': 'tweet', 'id': 'twitter_id'})
        
        return df
    
    def getDFFromTask (self, task, df):
        """ 
        Task 1 emulates previous humor detection tasks in which all ratings were averaged to provide mean classification and rating scores. 
        Task 2 aims to predict how offensive a text would be (for an average user) with values between 0 and 5. 
        """
        
        # Task 1a: predict if the text would be considered humorous (for an average user). This is a binary task.
        # -------------------------------------------------------------
        # We replace 0 with non-humor and 1 with humor
        if (task == '1a'):
            df['label'] = df['is_humor'].replace ([0, 1], ['non-humor', 'humor'])
            return df
        
        # Task 1b: if the text is classed as humorous, predict how humorous it is (for an average user). 
        # The values vary between 0 and 5.
        # -------------------------------------------------------------
        # We move the humor_rating column to the label and fill NaN with zeros
        if (task == '1b'):
            df['label'] = df['humor_rating']
            df['label'] = df['label'].fillna (0.0)
            return df;

        # Task 1c: if the text is classed as humorous, predict if the humor rating would be considered controversial, 
        #  i.e. the variance of the rating between annotators is higher than the median. This is a binary task.
        # -------------------------------------------------------------
        # We move the humor_controversy column to the label 
        if (task == '1c'):
        
            # It is possible to transform this problem into multiclass, to do it, just...
            df['label'] = df['is_humor'].replace ([0, 1], ['non-humor', 'humor'])
            df.loc[df['humor_controversy'] == 1, 'label'] = 'offensive'
        
            # It is possible to transform this problem into binary; however 
            # work has to be done to prevent the imbalance
            """
            df['label'] = df['humor_controversy'].replace ([0, 1], ['non-controversy', 'controversy'])
            df['label'] = df['label'].fillna ("non-controversy")
            """
            return df
        
        # Task 2a: predict how generally offensive a text is for users. 
        # This score was calculated regardless of whether the text is classed as humorous or offensive overall. 
        # -------------------------------------------------------------
        # Similar to 1b, we move the desired column to the labels, and then 
        # fill the missing values
        if (task == '2a'):
            df['label'] = df['offense_rating']
            df['label'] = df['label'].fillna (0.0)
            return df
