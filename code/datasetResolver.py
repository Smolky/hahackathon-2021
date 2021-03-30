"""
    DatasetResolver
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import config
import sys

from dataset import Dataset
from datasetPolitics import DatasetPolitics
from datasetHahackathon import DatasetHahackathon


class DatasetResolver ():
    """
    DatasetResolver
    """
    
    def get (self, dataset, options, refresh = False):

        # Default
        if not 'datasetClass' in options:
            return Dataset (dataset, options, refresh)

        # Super
        if (options['datasetClass'] == 'datasetPolitics'):
            return DatasetPolitics (dataset, options, refresh)
            
        elif (options['datasetClass'] == 'datasetHahackathon'):
            return DatasetHahackathon (dataset, options, refresh)

        else:
            return Dataset (dataset, options, refresh)
        


