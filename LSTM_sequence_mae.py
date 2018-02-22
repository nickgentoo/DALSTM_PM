"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import model_from_json
from math import sqrt
from keras.preprocessing import sequence
from keras import metrics
from load_dataset import load_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys


if len(sys.argv)<3:
    sys.exit("python LSTM_sequence.py n_neurons n_layers dataset")
n_neurons=int(sys.argv[1])
n_layers=int(sys.argv[2])
dataset=sys.argv[3]

# fix random seed for reproducibility
np.random.seed(7)

(X_train, y_train),(X_test, y_test)= load_dataset(dataset)

#normalize input data
#compute the normalization values only on training set
max=[0]*len(X_train[0][0])
for a1 in X_train:
    for s in a1:
        for i in xrange(len(s)):
            if s[i]>max[i]:
                max[i]=s[i]

for a1 in X_train:
    for s in a1:
        for i in xrange(len(s)):
             if(max[i]>0): #alcuni valori hanno massimo a zero
                s[i] =s[i]/ max[i]

for a1 in X_test:
    for s in a1:
        for i in xrange(len(s)):
            if (max[i] > 0):  # alcuni valori hanno massimo a zero
                s[i] = s[i] / max[i]


X_train = sequence.pad_sequences(X_train)
print("DEBUG: training shape",X_train.shape)
maxlen=X_train.shape[1]
X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1])
print("DEBUG: test shape",X_test.shape)



# create the model
model = Sequential()
if n_layers==1:
    model.add(LSTM(n_neurons, implementation=2,input_shape=(X_train.shape[1], X_train.shape[2]), recurrent_dropout=0.2))
    model.add(BatchNormalization())
else:
    for i in xrange(n_layers-1):
        model.add(LSTM(n_neurons, implementation=2, input_shape=(X_train.shape[1], X_train.shape[2]), recurrent_dropout=0.2,return_sequences=True))
        model.add(BatchNormalization())
    model.add(LSTM(n_neurons, implementation=2, recurrent_dropout=0.2))
    model.add(BatchNormalization())


#add output layer (regression)
model.add(Dense(1))

#compiling the model, creating the callbacks
model.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
print(model.summary())
early_stopping = EarlyStopping(patience=42)
model_checkpoint = ModelCheckpoint("model/model_"+dataset+"_"+str(n_neurons)+"_"+str(n_layers)+"_weights_best.h5", monitor='val_loss', verbose=0, save_best_only=True,
                                   mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

#saving model shape to file (saving model may be simpler with the new keras version)
model_json = model.to_json()
with open("model/model_"+dataset+"_"+str(n_neurons)+"_"+str(n_layers)+".json", "w") as json_file:
    json_file.write(model_json)

#train the model
model.fit(X_train, y_train, validation_split=0.2,  callbacks=[early_stopping, model_checkpoint,lr_reducer],epochs=500, batch_size=maxlen)

# Final evaluation of the model

#Create new model with saved architecture
#TODO it is possible to use the same model used for training, just loading the weights
testmodel = model_from_json(open("model/model_"+dataset+"_"+str(n_neurons)+"_"+str(n_layers)+".json").read())
#load saved weigths to the test model
testmodel.load_weights("model/model_"+dataset+"_"+str(n_neurons)+"_"+str(n_layers)+"_weights_best.h5")
# Compile model (required to make predictions)
testmodel.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
print("Created model and loaded weights from file")

#compute metrics on test set (same order as in the metrics list given to the compile method)
scores = model.evaluate(X_test, y_test, verbose=0)
print scores
print("Root Mean Squared Error: %.4f d MAE: %.4f d MAPE: %.4f%%" % (sqrt(scores[1]/((24.0*3600)**2)), scores[2]/(24.0*3600), scores[3]))
