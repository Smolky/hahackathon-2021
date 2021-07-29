# UMUTeam at SemEval-2021 Task 7: Detecting and Rating Humor and Offense with Linguistic Features and Word Embeddings
This project contains the source-code of the runs submitted by the UMUTeam at https://competitions.codalab.org/competitions/27446

More details regarding the task and the methods can be found at https://aclanthology.org/2021.semeval-1.152.pdf

## Abstract
In writing, humor is mainly based on figurative language in which words and expressions change their conventional meaning to refer to something without saying it directly. This flip in the meaning of the words prevents Natural Language Processing from revealing the real intention of a communication and, therefore, reduces the effectiveness of tasks such as Sentiment Analysis or Emotion Detection. In this manuscript we describe the participation of the UMUTeam in HaHackathon 2021, whose objective is to detect and rate humorous and controversial content. Our proposal is based on the combination of linguistic features with contextual and non-contextual word embeddings. We participate in all the proposed subtasks achieving our best result in the controversial humor subtask.


## Details
The source code is stored in the ```code``` folder. In the ```embeddings```folders there are symbolyc links to the pretrained word embeddings used. Due to size, you should download the ```glove.6b.300d.txt``` (https://nlp.stanford.edu/projects/glove/). The ```train```, ```dev```, and ```test``` splits are stored in the ```datasets``` folder. In the ```assets``` folders there are the features employed, the runs sent, and the hyperparameters results but not the models due to their filesize. You need to train the models or you can request me by email <joseantonio.garcia8@um.es>


## Citation
```
@inproceedings{garcia-diaz-valencia-garcia-2021-umuteam,
    title = "{UMUT}eam at {S}em{E}val-2021 Task 7: Detecting and Rating Humor and Offense with Linguistic Features and Word Embeddings",
    author = "Garc{\'\i}a-D{\'\i}az, Jos{\'e} Antonio  and
      Valencia-Garc{\'\i}a, Rafael",
    booktitle = "Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.semeval-1.152",
    pages = "1096--1101"
}
```
