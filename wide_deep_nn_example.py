import sys
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, train_test_split, PredefinedSplit
from keras.regularizers import l2
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Concatenate
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from random import shuffle, seed
from joblib import dump, load

sys.path.append ('./SourceCode')

from Load_Package import *
from Global_fun import *
from Config import Env_Config
Env_Config.fun_set_cwd(".")


data = pd.read_csv(fun_path_join(Env_Config.data_path,'customer_churn.csv'))
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
    x = data.groupby(['customerID'] + [fea])[fea].count().unstack(fea).reset_index()
    x.columns = [fea+'_'+x if x != 'customerID' else x for x in x.columns ]
    categorical_fea.append(x)

# mutiple outer join
num_fea =  [x for x in data.columns if x not in cat_var]
num_df = data.loc[:,num_fea]

# combine all the features together
categorical_fea.append(num_df)

all_fea = categorical_fea[0]
for fea in categorical_fea[1:]:
    all_fea = pd.merge(all_fea,fea,how='outer', on = 'customerID')

# Fill na
all_fea = all_fea.fillna(0)

# Create binary label
all_fea['Churn'] = np.where(all_fea['Churn']=='Yes', 1, 0)

# Create wide features
cat_fea = [x for x in all_fea.columns if any(fea in x for fea in cat_var)]
num_fea = [x for x in all_fea.columns if x not in cat_fea + ['customerID','Churn']]

# Create training and testing data
import random
random.seed(2020)

wide_fea= cat_fea
deep_fea= cat_fea + num_fea

ratio = 0.8
fea_ind = list(all_fea.index)
np.random.shuffle(fea_ind)

# index for training and testing data
train_idx = fea_ind[:int(0.8*len(fea_ind))]
test_idx = fea_ind[int(0.8*len(fea_ind))+1:]

# create training and testing data
# Training data for deep and wide model
Y_train = all_fea.loc[train_idx,['Churn']]
Y_test = all_fea.loc[test_idx,['Churn']]

X_train_wide = all_fea.loc[train_idx,wide_fea]
X_train_wide =X_train_wide.reset_index(drop=True)

X_test_wide = all_fea.loc[test_idx,wide_fea]
X_test_wide = X_test_wide.reset_index(drop=True)

# scale the numeric values
from sklearn.preprocessing import StandardScaler
X_train_deep = all_fea.loc[train_idx,deep_fea]
X_train_deep =X_train_deep.reset_index(drop=True)

X_test_deep = all_fea.loc[test_idx,deep_fea]
X_test_deep = X_test_deep.reset_index(drop=True)

std_scale = StandardScaler().fit(X_train_deep)
x_train_wide_norm = std_scale.transform(X_train_deep)
x_test_wide_norm = std_scale.transform(X_test_deep)

x_train_deep_norm = pd.DataFrame(X_train_deep.values, columns = list(X_train_deep.columns), index = list(X_train_deep.index) )
x_test_deep_norm = pd.DataFrame(X_test_deep.values, columns = list(X_test_deep.columns))

# create a final xtrain and xtest
x_train= pd.concat([x_train_deep_norm, X_train_wide], axis="columns")
x_test= pd.concat([x_test_deep_norm, X_test_wide], axis="columns")


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers import concatenate
from keras.utils import to_categorical

def deep_and_wide_model(x_train,y_train,activation,dropout_rate,loss_func,
                        learning_rate,metrics,wide_fea,optimiser,batch_size,
                        activation_output,random_seed,epochs):

    input_dim = x_train.shape[1]
    
    deep_fea = int(input_dim - wide_fea)
    
    print(f"input dim - {input_dim}, deep dim - {deep_fea}, wide dim - {wide_fea}")

    neuron_layer_1 = [deep_fea]
    neuron_layer_2 = [int(0.5*x) for x in neuron_layer_1]
    neurons = list(zip(neuron_layer_1,neuron_layer_2))[0]

   # first input model - wide
    wide_model = Input(shape=(wide_fea, ))

    # second input model - dense
    dense = Input(shape=(deep_fea, ))
    dense1 = Dense(units=neurons[0],
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01))(dense)
    normalisation1 = BatchNormalization()(dense1)
    activation1 = Activation(activation)(normalisation1)
    dropout1 = Dropout(dropout_rate, seed = random_seed)(activation1)

    dense2 = Dense(units=neurons[1],
           kernel_regularizer=l2(0.01),
           bias_regularizer=l2(0.01))(dropout1)
    normalisation2 = BatchNormalization()(dense2)
    activation2 = Activation(activation)(normalisation2)
    dropout2 = Dropout(dropout_rate, seed = random_seed)(activation2)

    deep_model = dropout2

    # merge input models
    merge = concatenate([deep_model, wide_model])

    y_binary = to_categorical(np.array(y_train))
    
    output_dim = y_binary.shape[1]
    
    output = Dense(output_dim, activation=activation_output)(merge)

    model = Model(inputs=[dense, wide_model], outputs=output)
    
    model.compile(optimizer=optimiser,loss=loss_func, metrics=metrics)
    
    deep_array = x_train[:,:deep_fea]
    print(f'deep array:{deep_array.shape}')
    wide_array = x_train[:,deep_fea::]
    print(f'wide array:{wide_array.shape}')
    
    model.fit([deep_array,wide_array], y_binary, epochs=epochs, batch_size=batch_size)
    
    return model

TRAIN = x_train.values
Y_TRAIN = Y_train.values
dw_model = deep_and_wide_model(x_train=TRAIN, y_train=Y_TRAIN, activation="relu", dropout_rate =0.5, 
                               loss_func = 'categorical_crossentropy',learning_rate=0.2,metrics =['accuracy'] ,
                               wide_fea = X_train_wide.shape[1],optimiser=Adam(),activation_output="softmax",
                               random_seed=2020,epochs=70,batch_size=32)

# predict 
fea_dim = x_train_deep_norm.shape[1]
TEST = x_test.fillna(0).values
prediction = dw_model.predict([TEST[:,:fea_dim],TEST[:,fea_dim::]])

pred_prob = pd.DataFrame(prediction, columns = ['neg','pos'])
churn_cust = list(np.where(pred_prob['pos']>pred_prob['neg'])[0])
pred_prob['pred'] = 0
pred_prob.loc[churn_cust,'pred'] = 1
pred_prob['truth'] = Y_test['Churn']

len(np.where(pred_prob['pred']==pred_prob['truth'])[0])/len(pred_prob)
