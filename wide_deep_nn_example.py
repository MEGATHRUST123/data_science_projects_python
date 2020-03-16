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

data = pd.read_csv('Customer_Churn.csv')

# Clean up data type
# Convert object fields into one hot encoding
categorical_fea = []

# data cleaning
data['TotalCharges'] = [0 if x == '' else x for x in [x.replace(' ','') for x in data['TotalCharges']]]
data['TotalCharges'] = data['TotalCharges'].astype('float64')

cat_var = [x for x in data.columns if data[x].dtype == 'object']
cat_var.pop(cat_var.index('customerID'))
cat_var.pop(cat_var.index('Churn'))

for fea in cat_var:
    print(f"Feature Engineering - {fea}")
    x = data.groupby(['customerID'] + [fea])[fea].count().unstack(fea).reset_index()
    x.columns = [fea+'_'+x if x != 'customerID' else x for x in x.columns ]
    print(x.head(2))
    categorical_fea.append(x)

# mutiple outer join
num_fea =  [x for x in data.columns if x not in cat_var]
num_df = data.loc[:,num_fea]

# combine all the features together
categorical_fea.append(num_df)

all_fea = categorical_fea[0]
for fea in categorical_fea[1:]:
    print(f' all fea - {all_fea.shape}')
    all_fea = pd.merge(all_fea,fea,how='outer', on = 'customerID')

# Fill na
all_fea = all_fea.fillna(0)

# Create binary label
all_fea['Churn'] = [1 if x == 'Yes' else 0 for x in all_fea['Churn']]

# model preprocessing 
# split data - 80/20 
import random
random.seed(2020)
ratio = 0.8
fea_ind = list(all_fea.index)
np.random.shuffle(fea_ind)
train_idx = fea_ind[:int(0.8*len(fea_ind))]
test_idx = fea_ind[int(0.8*len(fea_ind))+1:]

# create training and testing data
y_train = all_fea.loc[train_idx,['Churn']]
y_test = all_fea.loc[test_idx,['Churn']]
x_train = all_fea.loc[train_idx,list(set(all_fea.columns).difference(set(y_train.columns).union(['customerID'])))]
x_test = all_fea.loc[test_idx,list(y_train.columns)]

# train the model
batch_size_flag = 32
epoch_max = 70
dropout_rate = 0.4
metrics = ['accuracy']
cv = 1
glb_rand_seed = 2020
input_dim = x_train.shape

neuron_num_1st_layer = [int(0.1*input_dim), int(0.5*input_dim)]
neuron_num_2nd_layer = [int(x/2) for x in neuron_num_1st_layer]
neurons = list(zip(neuron_num_1st_layer, neuron_num_2nd_layer))
neurons = [list(x) for x in neurons]
optimizer = Adam()
activation_hidden = 'relu'
activation_output = 'softmax'
loss_fun = 'categorical_crossentropy'

# Create dense model
np.random.seed(glb_rand_seed)

dense = Sequential()    

#Layer 1 -- hidden               
dense.add(Dense(units = neurons[0][0], 
                input_shape = (input_dim[0],),                       
                kernel_initializer = 'random_uniform',                         
                kernel_regularizer = l2(0.01),
                bias_regularizer = l2(0.01)))    
dense.add(BatchNormalization())            
dense.add(Activation(activation_hidden))    
dense.add(Dropout(rate = dropout_rate, 
                    seed = glb_rand_seed))

# hidden layers - 2 hidden layers
for numnodes in neurons[1:]:
    dense.add(Dense(units = numnodes[0],                        
                    kernel_initializer = 'random_uniform',                         
                    kernel_regularizer = l2(0.01),
                    bias_regularizer = l2(0.01)))    
    dense.add(BatchNormalization())            
    dense.add(Activation(activation_hidden))    
    dense.add(Dropout(rate = dropout_rate, 
                        seed = glb_rand_seed))

# wide model
wide = Sequential() 
wide.add(Dense(units = neurons[0][0], 
                input_shape = (input_dim[0],),                         
                kernel_initializer = 'random_uniform',                         
                kernel_regularizer = l2(0.01),
                bias_regularizer = l2(0.01)))    
wide.add(BatchNormalization())            
wide.add(Activation(activation_hidden))    
wide.add(Dropout(rate = dropout_rate, 
                    seed = glb_rand_seed))

out_layer = Concatenate([dense.output, wide.output])
out_layer = (Dense(1, activation=activation_output)(out_layer)
model = Model(inputs=[dense.input, wide.input], outputs=model_concat)

# Compile model
model.compile(optimizer = Adam(),
                loss = loss_fun,
                metrics = glb_metrics)

# concatenate 2 models
# https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
