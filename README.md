# hahackathon-2021
This project contains the source-code of the runs submitted by the UMUTeam at https://competitions.codalab.org/competitions/27446

## Abstract
In writing, humor is mainly based on figurative language in which words and expressions change their conventional meaning to refer to something without saying it directly. This flip in the meaning of the words prevents Natural Language Processing from revealing the real intention of a communication and, therefore, reduces the effectiveness of tasks such as Sentiment Analysis or Emotion Detection. In this manuscript we describe the participation of the UMUTeam in HaHackathon 2021, whose objective is to detect and rate humorous and controversial content. Our proposal is based on the combination of linguistic features with contextual and non-contextual word embeddings. We participate in all the proposed subtasks achieving our best result in the controversial humor subtask.


## Details
The source code is stored in the ```code``` folder. In the ```embeddings```folders there are symbolyc links to the pretrained word embeddings used. Due to size, you should download the ```glove.6b.300d.txt``` (https://nlp.stanford.edu/projects/glove/). The ```train```, ```dev```, and ```test``` splits are stored in the ```datasets``` folder. In the ```assets``` folders there are the features employed, the runs sent, and the hyperparameters results but not the models due to their filesize. You need to train the models or you can request me by email <joseantonio.garcia8@um.es>