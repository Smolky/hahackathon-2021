'''
    Configuration of the datasets
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
'''

import sys
import netrc
import requests
import os
import random
import numpy as np
import pandas as pd


from requests.auth import AuthBase
from pathlib import Path



class PLNAuth (AuthBase):
    """
       PLNAuth
       
       We use a custom authentication because the default authentication
       was adding Basic authentication and do not allowed to manually 
       introduced the token
       
       @link https://requests.readthedocs.io/en/master/user/advanced/#custom-authentication
    """

    def __init__(self, username):
        self.username = username

    def __call__(self, r):
        r.headers['Authorization'] = self.username
        return r


# Seed
seed = 0 


# Configure main libraries for reproductibility
# Remember to use this seed in another random functions
os.environ['PYTHONHASHSEED'] = str (seed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
random.seed (seed)
np.random.seed (seed)


# @var certificate String
certificate = str (Path.home ()) + '/certificates/CA.pem'



def get_auth_token (api_endpoint, certificate):
    """
    @param api_endpoint
    @param certificate
    """

    # Read from the .netrc file in your home directory
    secrets = netrc.netrc ()
    email, account, password = secrets.authenticators ('collaborativehealth.inf.um.es')
    
    
    # @var reponse Response
    response = requests.post (
        api_endpoint + 'login', 
        json={'email': email, 'password': password}, 
        verify = certificate
    )
    
    
    # Transform to JSON
    response = response.json ()
    
    
    # ...
    return str (response['data']['token'])
    

# @var umutextstats_api_endpoint String
umutextstats_api_endpoint = 'php /home/rafa_pepe/umutextstats/api/umutextstats.php'


# @var umucorpusclassifier_api_endpoint String
umucorpusclassifier_api_endpoint = 'https://collaborativehealth.inf.um.es/corpusclassifier/api/'


# @var base_path String
base_path = Path (os.path.realpath (__file__)).parent.parent 


# @var directories Paths
directories = {
    'datasets': os.path.join (base_path, 'datasets'),
    'pretrained': os.path.join (base_path, 'embeddings', 'pretrained'),
    'sentence_embeddings': os.path.join (base_path, 'embeddings', 'sentences'),
    'assets': os.path.join (base_path, 'assets'),
    'cache': os.path.join (base_path, 'cache_dir'),
}


# @var pretrained_models 
pretrained_models = {
    'fasttext_original': {
        'binary': os.path.join (directories['pretrained'], 'cc.es.300.bin'),
        'vectors': os.path.join (directories['pretrained'], 'cc.es.300.vec'),
    },
    
    'word2vec': {
        'vectors': os.path.join (directories['pretrained'], 'word2vec-sbwc.txt')
    },
    
    'glove': {
        'vectors': os.path.join (directories['pretrained'], 'glove-sbwc.vec')
    },
    
    'fasttext_english': {
        'vectors': os.path.join (directories['pretrained'], 'cc.en.300.vec'),
    },
    
    'glove_english': {
        'vectors': os.path.join (directories['pretrained'], 'glove.6b.300d.txt'),
    }
}





# shared_max_size
shared_max_size = sys.maxsize


# @var shared_preprocessing Array Used for preprocessing tweets by preserving 
#                                 lettercase, mentions, emojis, ...
shared_preprocessing = [
    'lettercase', 
    'mentions', 
    'emojis', 
    'punctuation', 
    'digits', 
    'msg_language',
    'misspellings',
    'elongation'
]


# @var shared_umutextstats_preprocessing Array Used for extracting LF
shared_umutextstats_preprocessing = [
    "lettercase",
    "hyperlinks",
    "mentions",
    "emojis",
    "punctuation",
    "digits",
    "msg_language"
    "misspellings",
    "elongation",
    "preserve_multiple_spaces",
    "preserve_blank_lines"
]

# @var shared_postagging_preprocessing Array Used for extracting PoS
shared_postagging_preprocessing = [
    "lettercase",
    "punctuation"
]



# shared fields
shared_fields = ['twitter_id', 'twitter_created_at', 'user', 'tweet', 'label']


# shared label key
shared_label_key = 'label'


# shared shared_umutextstats
shared_umutextstats = 'default.xml'


# Train ratio (float or int, default=None)
shared_train_size = 0.8


# Validation ratio. This value will be used to split the train split again, achieving training 60 testing 20 eval 20
shared_train_val_size = 0.75


# IDs of the datasets
datasets = {
    "hahackathon": {
        "base": {
            'language': 'en',
            'datasetClass': 'datasetHahackathon', 
            'max': shared_max_size,
            'train_size': .8,
            'val_size': .2,
            'preprocessing': shared_preprocessing,
            'fields': shared_fields,
            'label_key': 'user',
            'umutextstats': {
                'dictionary': 'statistics.xml',
                'preprocessing': shared_umutextstats_preprocessing
            },
            'postagger': {
                'preprocessing': shared_postagging_preprocessing
            }
        },
    },
}