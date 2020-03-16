# Create wide and deep model
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, PredefinedSplit
from keras.regularizers import l2
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import DenseFeatures, concatenate
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from random import shuffle, seed
from joblib import dump, load


class wide_and_deep_model(object):
    
    def __init__(self, x_train, y_train, 
                    # change this : batch_size_flag: True -> False
                    batch_size_flag = False, 
                    epoch_max = 70, 
                    dropout_flag = False,
                    metrics = ['accuracy'], 
                    cv = 1,
                    rand_seed = glb_rand_seed):
        """Function: Initialize the class to train DNN
           Input:    1) 
                     2)       
                     x) cv. 1: use boot to tune model (fast), otherwise, cross validation (slow)               
           Output:              
        """
        self.x_train = x_train
        self.y_train = y_train 
        self.input_dim = x_train.shape[1]
        self.output_dim = y_train.shape[1]         
        self.batch_size_flag = batch_size_flag
        self.epoch_max = epoch_max # Use early stopping to decide whether to stop tuning        
        self.dropout_flag = dropout_flag 
        self.metrics = metrics
        if cv >= 1:
            self.cv = cv # 1: bootstrap, otherwise use CV to tune model parameters
        else:
            sys.exit("cv >= 1")
        
        self.rand_seed = rand_seed

        #Intialize the tuning parameters
        self.__fun_param_set()
    #End of Function "__init__"

    def __fun_param_set(self):
        """Function: Set parameters to train DNN based on input parameters
           Input:                                        
           Output:              
        """
        #set the number of neurons in hidden layers
        #layer-2 is half of layer-1
 
        neuron_num_1st_layer = [int(0.1*self.input_dim), int(0.5*self.input_dim)]
        neuron_num_2nd_layer = [int(x/2) for x in neuron_num_1st_layer]
        self.neurons = list(zip(neuron_num_1st_layer, neuron_num_2nd_layer))
        self.neurons = [list(x) for x in self.neurons]

        self.optimizer = Adam() 

        #Set activation function for hidden layer
        self.activation_hidden = 'relu'        

        #set activation function /loss function for output layer based on output dimensionality
        if self.output_dim > 1:
            self.activation_output = 'sigmoid' #multi-class multi-label classification
            self.loss_fun = 'binary_crossentropy'
        else:
            self.activation_output = 'softmax' #binary classfication
            self.loss_fun = 'categorical_crossentropy'

        #Set batch size
        if self.batch_size_flag == True:
            self.batch_size = [16, 32] #Tune batch size
        else:
            self.batch_size = [32] #fix batch size
        
        if self.dropout_flag == True:
            self.dropout_rate = [0.2, 0.4]
        else:
            self.dropout_rate = [0.4]     

        #split training data into training and validation (fast version of model training)
        if self.cv == 1:
            t_size = int(self.x_train.shape[0]*0.8)
            self.train_val_split = [-1]*t_size + [0]*(self.x_train.shape[0]-t_size)
            seed(self.rand_seed)
            shuffle(self.train_val_split)
            self.ps = PredefinedSplit(self.train_val_split)
        else:
            self.ps = self.cv
    #End of Function 'fun_param_set'
        
    
    def deep(self):

        # create model
        np.random.seed(self.glb_rand_seed)

        dense = Sequential()    

        #Layer 1 -- hidden               
        dense.add(Dense(units = self.neurons[0], 
                        input_shape = (self.input_dim,),                         
                        kernel_initializer = 'random_uniform',                         
                        kernel_regularizer = l2(0.01),
                        bias_regularizer = l2(0.01)))    
        dense.add(BatchNormalization())            
        dense.add(Activation(self.activation_hidden))    
        dense.add(Dropout(rate = self.dropout_rate, 
                            seed = self.glb_rand_seed))

        # hidden layers - 2 hidden layers
        for numnodes in self.neurons[1:]:
            dense.add(Dense(units = numnodes,                         
                            kernel_initializer = 'random_uniform',                         
                            kernel_regularizer = l2(0.01),
                            bias_regularizer = l2(0.01)))    
            dense.add(BatchNormalization())            
            dense.add(Activation(self.activation_hidden))    
            dense.add(Dropout(rate = self.dropout_rate, 
                                seed = self.glb_rand_seed))

        return dense
    
    def wide(self):
        
        wide = Sequential() 
        wide.add(Dense(units = self.neurons[0], 
                        input_shape = (self.input_dim,),                         
                        kernel_initializer = 'random_uniform',                         
                        kernel_regularizer = l2(0.01),
                        bias_regularizer = l2(0.01)))    
        wide.add(BatchNormalization())            
        wide.add(Activation(self.activation_hidden))    
        wide.add(Dropout(rate = self.dropout_rate, 
                            seed = self.glb_rand_seed))
        
        return wide
        
        
    def create_model(self):
        deep_model = self.deep()
        wide_model = self.wide()

        out_layer = concatenate([deep_model, wide_model])

        out_layer.add(Dense(units = self.output_dim,                         
                        kernel_initializer = 'random_uniform', 
                        activation = self.activation_output))
        
        # Compile model
        out_layer.compile(optimizer = Adam(),
                      loss = self.loss_fun,
                      metrics = self.glb_metrics)
        
        return out_layer
    
     def fun_train_model(self, reproducible = False):
            """Function: To train model
               Input:    1) 
                         2) 
               Output:  trained model
            """  
            #Not clear yet how to make model training reproducible
            #If gridSearchCV has only 1 CPU, it's reproducible
            np.random.seed(self.rand_seed)

            # create model     
            print('+'*30, " tuning param ", '-'*30)
            self.fun_print_param() #For debugging only

            model = KerasClassifier(build_fn = wide_and_deep_model.create_model, 
                                    input_dim = self.input_dim, 
                                    output_dim = self.output_dim, 
                                    activation_hidden = self.activation_hidden,
                                    activation_output = self.activation_output,
                                    loss_fun = self.loss_fun,                                 
                                    verbose=0)

            #set parameter grid        
            param_grid = dict(neurons = self.neurons, 
                                batch_size = self.batch_size, 
                                dropout_rate = self.dropout_rate)

            if reproducible == True:
                # change this: from 1 to 2 for parallel computing
                job_num = 3
            else:
                job_num = -1 #use all processes

            grid = GridSearchCV(estimator = model, 
                                param_grid = param_grid, 
                                n_jobs = job_num,                             
                                cv = self.ps)

            # how to calculate accuracy
            # why accuracy and not loss/recall
            earlystop_callback = EarlyStopping(monitor = 'accuracy', 
                                                mode = 'max', 
                                                patience = 1)
            # change this: verbose = 2, one epoch, one line
            grid_result = grid.fit(X = self.x_train, 
                                    y = self.y_train, 
                                    epochs = self.epoch_max, 
                                    callbacks=[earlystop_callback],
                                    verbose = 2)
            # summarize results        
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

            return grid_result
