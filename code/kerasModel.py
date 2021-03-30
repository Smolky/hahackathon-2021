"""
    Keras and Talos por hyper-parameter tunning
    
    A way to create dinamically a Keras Model. It can be used with 
    params for hyper-parameter optimisation, or to generate dinamically 
    one combination
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

# Import Libs
import utils
import tensorflow
import sys
import talos

from tensorflow import keras
from keras import backend as K


def root_mean_squared_error (y_true, y_pred):
    """
    Returns the RMSE
    
    @link https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras
    @param y_true
    @param y_pred
    """
    return K.sqrt (K.mean (K.square (y_pred - y_true)))



def create (x_train, y_train, x_val, y_val, params):
    
    """
    Create a Keras Model for linguistic features and word embeddings
    
    This model can be used with Talos or to create a specific Keras 
    model
    
    @param x_train Features
    @param y_train Labels
    @param x_val Features for validation 
    @param y_val Labels for validation
    @param params Dict
    
    Constructor
    """
        
    # Extract variables from params for better readability
    name = params['name']
    number_of_classes = params['number_of_classes']
    first_neuron = params['first_neuron']
    number_of_layers = params['number_of_layers']
    shape = params['shape']
    architecture = params['we_architecture']
    dropout_range = params['dropout']
    trainable_embedding_layer = params['trainable']
    maxlen = params['maxlen']
    pretrained_embeddings = params['pretrained_embeddings']
    optimizer = params['optimizer']
    lr = params['lr']
    batch_size = params['batch_size']
    epochs = params['epochs']
    kernel_size = params['kernel_size']
    features = params['features']
    patience = params['patience']
    task_type = params['task_type']
    tokenizer = params['tokenizer']
    dataset = params['dataset']
    
    
    # Get the full tokenizer size
    vocab_size = len (tokenizer.word_index) + 1

    
    # @var pretrained_models from the tokenizer. 
    custom_embeddings = {
        'fastText': utils.get_embedding_matrix ('fasttext_english', tokenizer, dataset, name),
        'glove': utils.get_embedding_matrix ('glove_english', tokenizer, dataset, name)
    }
    
    
    # Helpers to indicate what kind of features we are using
    has_lf = features in ['lf', 'lf+we']
    has_we = features in ['we', 'lf+we']
    
    
    # Determine configuration according to the task type
    if (task_type == 'classification'):
    
        # Determine if the task is binary or multi-class
        is_binary = number_of_classes == 1
        
        
        # @var last_activation_layer String Get the last activation layer based on the number of classes
        # @link https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
        last_activation_layer = 'sigmoid' if is_binary else 'softmax'
        
        
        # @var loss_function String Depends if multiclass or binary
        # @link https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        loss_function = 'binary_crossentropy' if is_binary else 'categorical_crossentropy'
        
        
        # @var number_of_neurons_in_the_last_layer int
        number_of_classes_in_the_last_layer = number_of_classes
        
        
        # @var metric Select the metric according to the problem
        metric = keras.metrics.BinaryAccuracy (name = "accuracy") if is_binary else keras.metrics.CategoricalAccuracy (name = 'accuracy')
        
    
    # Regression problems
    else:
        
        # Get the last activation layer
        # @link https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
        # @todo set by param
        last_activation_layer = 'linear'
        
        
        # Define loss function as RMSE
        # @todo set by param
        loss_function = root_mean_squared_error
    
    
        # @var metric Select the metric according to the problem
        metric = tensorflow.keras.metrics.RootMeanSquaredError (name='rmse')
        
        
        # @var number_of_neurons_in_the_last_layer int 
        number_of_classes_in_the_last_layer = 1
        
    
    # @var neurons_per_layer List Contains a list of the neurons per layer according to 
    #                             different shapes
    neurons_per_layer = utils.get_neurons_per_layer (shape, number_of_layers, first_neuron)
    
    
    # Define the input layers
    # 1) If we have word embeddings
    if has_we:
    
        # @var layer_we_input Main embedding layer
        layer_we_input = keras.layers.Input (shape = (maxlen,))
    
    
        # @var embedding_dim int Get the embedding dimension
        # @note We use 300 if no pretrained dimension was supplied because is the 
        #                  same number as the majority of embeddings we used. 
        # @todo Allow to parametrise this number
        embedding_dim = custom_embeddings[pretrained_embeddings].shape[1] if pretrained_embeddings != 'none' else 300 
        
        
        # @var weights multidimensional List Get the weights from the pretrained word embeddings used in this task
        #                                    If not set, use the default value
        weights = [custom_embeddings[pretrained_embeddings]] if pretrained_embeddings != 'none' else None
        
        
        # Input for word embeddings require a embedding layer with the weights
        # In Keras, each layer has a parameter called “trainable”. For freezing the weights 
        # of a particular layer, we should set this parameter to False, indicating that this 
        # layer should not be trained. 
        # @link https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models
        layer_we = keras.layers.Embedding (
            input_dim = vocab_size, 
            output_dim = embedding_dim, 
            weights = weights, 
            input_length = maxlen,
            trainable = trainable_embedding_layer
        )(layer_we_input)

    
        # Generate word embedding architecture
        # Some notes about the pooling layers
        
        # GlobalMaxPool1D
        # @link https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
        
        # SpatialDropout1D
        # @link https://stackoverflow.com/questions/50393666/how-to-understand-spatialdropout1d-and-when-to-use-it
        
        # @todo. Param recurrent dropout different from dropout
        
        # Multilayer perceptron. This layer will be connected to the rest of the MLP
        if (architecture == 'dense'):
            layer_we = keras.layers.GlobalMaxPool1D ()(layer_we)
        
        # Convolutional neuronal network
        if (architecture == 'cnn'):
            layer_we = keras.layers.SpatialDropout1D (dropout_range)(layer_we)
            layer_we = keras.layers.Conv1D (neurons_per_layer[0], kernel_size, activation = params['activation'])(layer_we)
            layer_we = keras.layers.GlobalMaxPool1D ()(layer_we)

        # LSTM
        if (architecture == 'lstm'):
            layer_we = keras.layers.SpatialDropout1D (dropout_range)(layer_we)
            layer_we = keras.layers.LSTM (neurons_per_layer[0], dropout = dropout_range, recurrent_dropout = dropout_range)(layer_we)

        # GRU
        if (architecture == 'gru'):
            layer_we = keras.layers.SpatialDropout1D (dropout_range)(layer_we)
            layer_we = keras.layers.GRU (neurons_per_layer[0], dropout = dropout_range, return_sequences=False)(layer_we)
            
        # BiLSTM
        if (architecture == 'bilstm'):
            layer_we = keras.layers.SpatialDropout1D (dropout_range)(layer_we)
            layer_we = keras.layers.Bidirectional (keras.layers.LSTM (neurons_per_layer[0], dropout = dropout_range, recurrent_dropout = dropout_range))(layer_we)

        # BiGRU
        if (architecture == 'bigru'):
            layer_we = keras.layers.SpatialDropout1D (dropout_range)(layer_we)
            layer_we = keras.layers.Bidirectional (keras.layers.GRU (neurons_per_layer[0], dropout = dropout_range, return_sequences=False))(layer_we)
    
    
    # Define the input layers
    # @var layer_lf_input Main lf layer
    if has_lf:
        layer_lf_input = keras.layers.Input (shape = (x_train[1].shape[1],))
    
    
    # Merge layers if needed
    if has_lf and has_we:
        layer_merged = keras.layers.concatenate ([layer_we, layer_lf_input])
    
    elif has_lf:
        layer_merged = layer_lf_input
        
    else:
        layer_merged = layer_we
    
    
    # Concatenate with the deep-learning network
    x = layer_merged
    for i in range (number_of_layers):
        x = keras.layers.Dense (neurons_per_layer[i])(x)
        if (dropout_range):
            x = keras.layers.Dropout (dropout_range)(x)
    layer_merged = x
    
    
    # Inputs
    if has_lf and has_we:
        inputs = [layer_we_input, layer_lf_input]
        x_train = x_train
        x_val = x_val
    
    elif has_lf:
        inputs = [layer_lf_input]
        x_train = x_train[1]
        x_val = x_val[1]
        
    else:
        inputs = [layer_we_input]
        x_train = x_train[0]
        x_val = x_val[0]
    
    
    # Outputs
    outputs = keras.layers.Dense (number_of_classes_in_the_last_layer, activation = last_activation_layer)(layer_merged)
    
    
    # Create model
    model = keras.models.Model (inputs = inputs, outputs = outputs, name = params['name'])
    
    
    # @var Optimizer
    optimizer = optimizer (lr = talos.utils.lr_normalizer (lr, optimizer))
    

    # Compile model
    model.compile (optimizer = optimizer, loss = loss_function, metrics = [metric])
    
    
    # @var early_stopping Early Stopping callback
    early_stopping = tensorflow.keras.callbacks.EarlyStopping (
        monitor = 'val_loss' if task_type == 'classification' else 'val_rmse', 
        patience = patience,
        restore_best_weights = True
    )
    
    
    # Fit model
    history = model.fit (
        x = x_train, 
        y = y_train,
        validation_data = (x_val, y_val),
        batch_size = batch_size,
        epochs = epochs,
        callbacks = [early_stopping]
    )
    
    
    """
    for prediction in model.predict (x_val):
        print (prediction.shape)
    """
    
    
    # finally we have to make sure that history object and model are returned
    return history, model
    
    
def main ():
    """ To use from command line """
    
    
    
if __name__ == "__main__":
    main ()