#########################################################################################################################################
# Project     : Automation of Model Training
#
# Coding      : CAO Jianneng
#
# Date        : Since 2019-12-21
#
# Note        : Determining the Number of Hidden Layers
#   0 - Only capable of representing linear separable functions or decisions.
#   1 - Can approximate any function that contains a continuous mapping
#       from one finite space to another.
#   2 - Can represent an arbitrary decision boundary to arbitrary accuracy
#       with rational activation functions and can approximate any smooth
#       mapping to any accuracy.
#
# Questions   : Not clear how to send metrics as input paramter to build estimator
#
# Description : Automate the training of a dense neutral network
#               1) 
#               2) 
#########################################################################################################################################


from Model.model_config_deep_wide import *
from Global_fun import *

from sklearn.metrics import recall_score

glb_metrics = ['accuracy']
glb_rand_seed = 2020

class cls_auto_DNN(object):
    """Class to automate the trainng of DNN. It includes functions:
       1) Set the parameters for model tuning
       2) Model training
       3) Prediction
       4) Evaluation
    """
    def __init__(self, x_train, y_train, x_test, wide_fea_index,
                    # change this : batch_size_flag: True -> False
                    batch_size_flag = False, 
                    epoch_max = 5, 
                    dropout_flag = False,
                    metrics = ['accuracy'], 
                    cv = 1,
                    mode = 'all',
                    rand_seed = glb_rand_seed):
        """Function: Initialize the class to train DNN
           Input:    1) 
                     2)       
                     x) cv. 1: use boot to tune model (fast), otherwise, cross validation (slow)               
           Output:              
        """
        self.x_train = x_train
        self.y_train = y_train 
        self.x_test = x_test
        self.wide_fea_index = wide_fea_index # needed for wide model
        
        self.input_dim = x_train.shape[1]
        self.output_dim = y_train.shape[1]    
        self.wide_dim = len(wide_fea_index) # number of features for wide model
        
        if mode == 'all':
            self.dense_dim = x_train.shape[1]
        else:
            self.dense_dim = x_train.shape[1]- len(wide_fea_index)
        
        self.batch_size_flag = batch_size_flag
        self.epoch_max = epoch_max # Use early stopping to decide whether to stop tuning        
        self.dropout_flag = dropout_flag 
        self.metrics = metrics
        self.mode = mode
        
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
        neuron_num_1st_layer = [int(0.1*self.dense_dim)]
       
        #NOTE. if memory is not enough
        """
        # Change this: reduce the number of neurons in hidden layer one 
        parameter = 0.2
        neuron_num_1st_layer = [self.input_dim, int(1.5*self.input_dim), 2*self.input_dim]
        neuron_num_1st_layer = [int(parameter*x) for x in neuron_num_1st_layer]
        """
        neuron_num_2nd_layer = [int(x/2) for x in neuron_num_1st_layer]
        self.neurons = list(zip(neuron_num_1st_layer, neuron_num_2nd_layer))
        self.neurons = [list(x) for x in self.neurons]

        self.optimizer = Adam() # By default we use Adam, and not tune learning rate

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
            self.batch_size = 32 #fix batch size
        
        if self.dropout_flag == True:
            self.dropout_rate = [0.2, 0.4]
        else:
            self.dropout_rate = 0.5
            
        if self.wide_dim != 0:
            # x_train is a numpy array - need to split by index
            # create training data for deep and wide
            # x train - dense features + wide features
            if self.mode == 'all':
                # all training + wide
                wide_fea = self.x_train[:,self.wide_fea_index]
                self.x_train_new = [self.x_train] + [wide_fea]
                
            else:
                dense_fea_index = list(range(self.x_train.shape[1]))
                # sort the index
                dense_fea_index = list(np.unique([x for x in dense_fea_index if x not in self.wide_fea_index]))
                dense_fea = self.x_train[:,dense_fea_index]
                
                print(f'dense_fea: {dense_fea.shape}')
                wide_fea = self.x_train[:,self.wide_fea_index]
                self.x_train_new = [dense_fea] + [wide_fea]
        else:
            sys.exit('Wide Features are not provided')
            

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

    def fun_print_param(self):
        """Function: To print out parameters to tune model. Debugging only
           Input:                                        
           Output:              
        """     
        print("neurons: ", self.neurons)
        print("lst layer neurons: ", self.neurons[0][0])
        print("2nd layer neurons: ", self.neurons[0][1])
        
        print("optimizer: adam" )        
        print("hidden activation: ", self.activation_hidden)
        print("output activation: ", self.activation_output)
        print("loss function: ", self.loss_fun)
        print("batch size: ", self.batch_size)    
        print("dropout rate: ", self.dropout_rate)
        print("metrics: ", self.metrics)
        print("cv: ", self.cv)
        if self.cv == 1:
            for train_index, validation_index in self.ps.split():
                print("training: ", len(train_index), "; validation: ", len(validation_index))
    #End of Function 'fun_print_param'
    
    @staticmethod
    def __fun_create_dw_model(neurons, dropout_rate, 
                        input_dim, output_dim, wide_dim,dense_dim,
                        activation_hidden, activation_output,
                        # eval_metrics,
                        loss_fun):
        """Function: To create model estimator
            Note. Not clear how to set metrics as an input parameter
            Input:    1) neurons. A pair: 1st = #neurons in 1st hidden layer. 
                                            2st = #neurons in 2nd hidden layer  
                    2) dropout_rate.                  
            Output:  estimator
        """  
        print('+'*30, " parameters inside Function fun_create_model ", '-'*30)
        print("input_dim = {}, output_dim = {}, activation_hidden = {}, activation_output = {}, loss_fun = {}" \
                    .format(input_dim, output_dim, activation_hidden, activation_output, loss_fun))    

        # create model
        np.random.seed(glb_rand_seed)

        print(f"input dim - {input_dim}, deep dim - {dense_dim}, wide dim - {wide_dim}, output dim - {output_dim}")

       # first input model - wide
        wide_model = Input(shape=(wide_dim, ))

        # second input model - dense
        dense = Input(shape=(dense_dim, ))
        dense1 = Dense(units=neurons[0][0],
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01))(dense)
        normalisation1 = BatchNormalization()(dense1)
        activation1 = Activation(activation_hidden)(normalisation1)
        dropout1 = Dropout(dropout_rate, seed = glb_rand_seed)(activation1)

        dense2 = Dense(units=neurons[0][1],
                       kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01))(dropout1)
        normalisation2 = BatchNormalization()(dense2)
        activation2 = Activation(activation_hidden)(normalisation2)
        dropout2 = Dropout(dropout_rate, seed = glb_rand_seed)(activation2)

        deep_model = dropout2

        # merge input models
        merge = concatenate([deep_model, wide_model])
        
        print(output_dim)

        output = Dense(output_dim,kernel_initializer="random_uniform",activation="sigmoid")(merge)

        model = Model(inputs= [dense] + [wide_model], outputs=output)

        model.compile(optimizer = Adam(),
              loss = loss_fun,
              metrics = glb_metrics)
        
        return model
    #End of Function 'fun_create_model'
    
        
    def fun_train_pred(self, reproducible = False):
        
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

        model = cls_auto_DNN.__fun_create_dw_model(neurons = self.neurons,dropout_rate = self.dropout_rate,
                                input_dim = self.input_dim, output_dim = self.output_dim, wide_dim = self.wide_dim,
                                dense_dim = self.dense_dim, activation_hidden = self.activation_hidden,
                                activation_output = self.activation_output, loss_fun = self.loss_fun)
        
        
        # change this: verbose = 2, one epoch, one line
        model.fit(x = self.x_train_new,  y = self.y_train,  epochs = self.epoch_max, batch_size=self.batch_size)
        
        model.save("wide_and_deep_k.h5")
        
        pred = model.predict(self.x_test)

        return pred, model
    
    #End of Function 'fun_train_model'
    @classmethod
    def fun_pred(cls, grid_result, x_test, type = 'none'): 
        
        if type == 'prob':
            pred = grid_result.predict_proba(x_test)
        else:
            pred = grid_result.predict(x_test)
        return pred
    #End of Function 'fun_pred'
    
    @classmethod
    def remove_zero_row(cls, pred: np.array, test: np.array):
        # Remove rows with all zero values

        pred_non_zero = pred[~np.all(test == 0, axis=1)]
        test_non_zero = test[~np.all(test == 0, axis=1)]

        return pred_non_zero, test_non_zero

    @classmethod
    def change_top_n_0_1(cls, pred: np.array, top_n: int):
        """ Change top n values to 1, and the rest to 0 """

        n_row, n_col = pred.shape
        index_above = n_col - top_n
        pred_0_1 = np.copy(pred)

        pred_0_1[np.arange(n_row)[:, None],
                 np.argsort(pred_0_1)[:, index_above:n_col]] = 1

        pred_0_1[pred_0_1 < 1] = 0

        return pred_0_1

    @classmethod
    def get_recall(cls, pred: np.array, test: np.array, top_n: int=3, avg_method=None):
        """ Generate recall score """
        pred0_1 = cls.change_top_n_0_1(pred, top_n=top_n)

        return recall_score(test, pred0_1, average=avg_method)
        #return cls.fun_compute_recall(pred = pred0_1, truth = test)
    
    @classmethod
    def fun_compute_recall(cls, pred, truth):
        """
        """
        rslt = np.zeros(pred.shape[1])
        for i in list(range(pred.shape[1])):
            x = pred[:,i]
            y = truth[:,i]
            inter = [idx for idx in list(range(len(x))) if (x[idx]==1) and (y[idx]==1)] # TP count
            rslt[i] = len(inter)/np.sum(y)
        
        return rslt        
    #End of Function 'fun_compute_recall'
   
    
#End of class 'cls_auto_DNN
