#Created by Sai Teja Mummadi, Computer Science, Michigan Technological University (Houghton, Mi)
#Training and Saving hybrid (CNN+ML) models
#Input: Xtrain.npy,ytrain.npy test.npy
#Output: Accuracy_Table.csv, Predictions.csv

## Libraries
#Libraries for File operations
import os
import numpy as np
import pandas as pd
import pickle

#Libraries for Visualization
import matplotlib.pyplot as plt


#Libraries for Neural Networks
import tensorflow as tf
import keras
from keras import regularizers
from keras import layers
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

#Libraries for Machine Learning
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

import datetime
import pytz
central = pytz.timezone('US/Eastern')
now = datetime.datetime.now(central)
date_string = now.strftime("%Y-%m-%d-%H-%M-%S")

np.set_printoptions(suppress=True)

##Model Building pipeline
traindataname = input('Enter the filename of train data eg: Xtraindata.npy: ')
ytraindataname = input('Enter the filename of y train data eg: ytraindata.npy: ')
traindata = np.load('Finaldata/'+traindataname)
ytraindata = np.load('Finaldata/'+ytraindataname)

testfilenumber = int(input('Enter the number of test files: '))
testfiles = []
testfilenames = []
for i in range(testfilenumber):
    name  = input('Enter the test file '+str(i+1)+' name Eg:Testfilename.npy:')
    testfilenames.append(name)
    testfiles.append(np.load('Finaldata/'+name))

X_train, X_test, y_train, y_test = train_test_split(traindata, ytraindata, train_size=0.80, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.50, random_state=66)

trainshape = traindata.shape
appropriate_shape = trainshape[1:]

newpath = r'Savedmodels_cnn' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

#Changable Hyperparameters for Encoder model

batchsize = 64  #Other batch sizes 128, 256, 512, 1024, etc.
EPOCH = 100     #Other epochs 50, 80, 100, 150, etc.
learningrate = 0.00003 #Other learning rates are 0.1, 0.001, 0.0001, 0.00001 etc.

#Optimizer for Encoder
opt = tf.keras.optimizers.RMSprop(learning_rate=learningrate)
# opt = tf.keras.optimizers.Adam(learning_rate=learningrate)
# opt = tf.keras.optimizers.SGD(learning_rate=learningrate)
# opt = tf.keras.optimizers.Adagrad(learning_rate=learningrate)
# opt = tf.keras.optimizers.Adadelta(learning_rate=learningrate)


#Activation functions
activation = 'relu'
# activation = tf.keras.activations.tanh()
# activation = tf.keras.activations.selu()
modelnames = []


#Different Loss Functions
los1 = tf.keras.losses.mean_squared_logarithmic_error
los2 = tf.keras.losses.MeanSquaredError()
los3 = 'binary_crossentropy'
los4 = tf.keras.losses.MeanAbsoluteError()
los5 = tf.keras.losses.Hinge()
los6 = tf.keras.losses.Poisson()
los7 = tf.keras.losses.Huber()
los8 = tf.keras.losses.LogCosh()

losdict = {
    'mean_squared_logarithmic_error' : los1,
    'MeanSquaredError' : los2,
    'binary_crossentropy' : los3,
    'MeanAbsoluteError' : los4,
    'Hinge_Loss' : los5,
    'Poisson_Loss':los6,
    'Huber_Loss' : los7,
    'LogCost_Loss':los8
}

#Change the number of Kernels
C1 = 32  #Other sizes could be 16, 32, 64, 128, 256 ...
C2 = 32  #Other sizes could be 16, 32, 64, 128, 256 ...
C3 = 64  #Other sizes could be 16, 32, 64, 128, 256 ...
C4 = 64  #Other sizes could be 16, 32, 64, 128, 256 ...

#Change Kernel size
ksize = 3 #Other size could be 3,4,5 and not more than 6

def export_training_accimage(history, moname):
    #Plotting Encoder training
    tloss1=history.history['loss']
    tacc1=history.history['accuracy']

    vloss1=history.history['val_loss']
    vacc1=history.history['val_accuracy']

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,4))
    ax2.plot(tloss1, color='red')
    ax2.plot(vloss1, color='green')

    ax1.plot(tacc1,color='red')
    ax1.plot(vacc1,color='green')
    fig.suptitle('Accuracy and Loss of CNN model')

    # ax1.set(facecolor = "#D7DFF5")
    # ax2.set(facecolor = "#D7DFF5")
    ax1.set(xlabel='Number of Epochs', ylabel='Accuracy')
    ax2.set(xlabel='Number of Epochs', ylabel='Loss')

    ax1.legend(['Train accuracy', 'Validation accuracy'], loc='upper left')
    ax2.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    ax1.grid()
    ax2.grid()

    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/CNN_'+moname+'.png')
    plt.close()
#Accuracy function for test data
def printacc(model, data):
    predictions = model.predict(data)
    predictions = np.round(abs(predictions))
    positivelabels = (predictions==1).sum()
    negativelabels = (predictions==0).sum()
#    print('Negative:',negativelabels,'Positive:',positivelabels)
    rate = positivelabels/(negativelabels+positivelabels)
    return rate


def export_predictions_nn(model, data,modname,testfilenames=testfilenames):
    for i,file in enumerate(data):
        testfilenam = testfilenames[i]
        outputfile = pd.read_csv('Finaldata/Predictions/predvals'+testfilenam[:-4]+'.csv')
        outputfile[modname] = model.predict(file)
        outputfile.to_csv('Finaldata/Predictions/predvals'+testfilenam[:-4]+'.csv',index=False)
        
    

def getmodel():    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3,activation = 'relu', input_shape=appropriate_shape  ))
    model.add(BatchNormalization())
    model.add(keras.layers.Conv2D(64, activation = 'relu',kernel_size=3))
    model.add(BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

# cnn = getmodel()
accuracies = []

for i in losdict:
    cnnmodel = getmodel()
    cnnmodel.compile(optimizer=opt, loss=losdict[i],metrics='accuracy')
    history = cnnmodel.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = 100,epochs=EPOCH)
    export_training_accimage(history, i)
    pickle.dump(cnnmodel, open('Savedmodels_cnn/CNN_'+i+'.sav', 'wb'))
    #Results of Encodermodel


    val_acc = cnnmodel.predict(X_val)
    val_acc[val_acc>0.5]=1
    val_acc[val_acc<0.5]=0
    #print(accuracy_score(val_acc,y_val))
    accuracies.append(accuracy_score(val_acc,y_val))

    test_acc = cnnmodel.predict(X_test)
    test_acc[test_acc>0.5]=1
    test_acc[test_acc<0.5]=0
    #print(accuracy_score(test_acc,y_test))
    accuracies.append(accuracy_score(test_acc,y_test))

    for data in testfiles:
        accuracies.append(printacc(cnnmodel, data))

    print(classification_report(test_acc,y_test, target_names=['Regulation','No Regulation']))

    name_to_print = 'CNN_'+i

    if testfilenumber !=0:
        export_predictions_nn(cnnmodel, testfiles, name_to_print)

    modelnames.append(name_to_print)
   
accuracies2 = np.array(accuracies)
accuracies2 = accuracies2.reshape(len(modelnames),testfilenumber+2)
headernames = ['Validation','Testing']
for i in range(testfilenumber):
    testfilenam = testfilenames[i]
    headernames.append(testfilenam[:-4])

accuracy_df = pd.DataFrame(accuracies2)
accuracy_df = accuracy_df.set_index([modelnames])
accuracy_df.columns = headernames


accuracy_df.to_csv('Savedmodels_cnn/accuracy_table'+date_string+'.csv')
print('Accuracy values are saved in same folder and Predictions are saved at Finaldata/Predictions')

motiffilenumber = int(input('Number of motif file available: '))
motifcount = 0
for i in range(testfilenumber):
    if motifcount == motiffilenumber:
        break
    flag = input('Is motif file available for '+testfilenames[i]+'? Enter y or n:')
    if(flag == 'y'):
        motiffilename =  input('Enter the Motif file name or path: ')
        motifdata = pd.read_csv(motiffilename)
        newtestfilename = testfilenames[i]
        preddata = pd.read_csv('Finaldata/Predictions/predvals'+newtestfilename[:-4]+'.csv')
        result = pd.merge(preddata, motifdata,how="left",  on=["TF","Gene"])
        result.to_csv('Finaldata/Predictions/predvals'+newtestfilename[:-4]+'.csv',index=False)
        motifcount += 1
    else:
        pass
print('The Final output files are saved at Finaldata/Predictions/')    
 
