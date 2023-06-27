
import os
import numpy as np
import pandas as pd
#Libraries for Visualization
import matplotlib.pyplot as plt
import pickle
import datetime
import pytz

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

np.set_printoptions(suppress=True)

##Model Building pipeline
traindataname = input('Enter the filename of train data eg: Xtraindata.npy: ')
ytraindataname = input('Enter the filename of y train data eg: ytraindata.npy: ')
# traindataname = 'Xtraindata.npy'
# ytraindataname = 'ytraindata.npy'
traindata = np.load('Finaldata/'+traindataname)
ytraindata = np.load('Finaldata/'+ytraindataname)



testfilenumber = int(input('Enter the number of test files: '))
testfiles = []
testfilenames = []
for i in range(testfilenumber):
    name  = input('Enter the test file '+str(i+1)+' name Eg:Testfilename.npy:')
    # name = testdata[i]
    testfilenames.append(name)
    testfiles.append(np.load('Finaldata/'+name))

X_train, X_test, y_train, y_test = train_test_split(traindata, ytraindata, train_size=0.80, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.50, random_state=66)

trainshape = traindata.shape



newpath = r'Savedmodels_ANN' 
if not os.path.exists(newpath):
    os.makedirs(newpath)


 

        

accuracies = []


#Transforming Training data for ML models
X_train_new = np.reshape(X_train, (X_train.shape[0], -1))

X_val_new = np.reshape(X_val, (X_val.shape[0], -1))
X_test_new = np.reshape(X_test, (X_test.shape[0], -1))

testfiles_new = []
for file in testfiles:
    testfiles_new.append(np.reshape(file, (file.shape[0], -1)))

data_new = [X_val_new, y_val, X_test_new, y_test,testfiles_new]
modelnames = []


#Changable Hyperparameters for Encoder model

batchsize = 100  #Other batch sizes 128, 256, 512, 1024, etc.
EPOCH = 75     #Other epochs 50, 80, 100, 150, etc.
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
    'LogCosh_Loss':los8
}


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
    fig.suptitle('Accuracy and Loss of ANN model')

    # ax1.set(facecolor = "#D7DFF5")
    # ax2.set(facecolor = "#D7DFF5")
    ax1.set(xlabel='Number of Epochs', ylabel='Accuracy')
    ax2.set(xlabel='Number of Epochs', ylabel='Loss')

    ax1.legend(['Train accuracy', 'Validation accuracy'], loc='upper left')
    ax2.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    ax1.grid()
    ax2.grid()

    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/ANN_'+moname+'.png')
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
        
    




# Get the current date
central = pytz.timezone('US/Eastern')

now = datetime.datetime.now(central)

# Format the date as a string using strftime()
date_string = now.strftime("%Y-%m-%d-%H-%M-%S")


###################################################
#Neural Network
def getmodel():    
    print('Running Neural Network model')
    inputlength = len(X_train_new[1])
    annmodel = Sequential()
    annmodel.add(Dense(256, activation = activation,input_shape=(inputlength,)))
    annmodel.add(BatchNormalization())
    annmodel.add(Dropout(0.1))
#     annmodel.add(Dense(4000, activation = activation, kernel_initializer = 'he_uniform'))
#     annmodel.add(BatchNormalization())
#     annmodel.add(Dense(2000, activation = activation, kernel_initializer = 'he_uniform'))
#     annmodel.add(BatchNormalization())
#     annmodel.add(Dropout(0.1))
    annmodel.add(Dense(128, activation = activation))
    annmodel.add(BatchNormalization())
    annmodel.add(Dropout(0.1))
    annmodel.add(Dense(1, activation = 'sigmoid'))
    return annmodel



accuracies = []
histories_new = []
labels = []

for i in losdict:
    annmodel = getmodel()
    annmodel.compile(optimizer=opt, loss=losdict[i],metrics='accuracy')
    history = annmodel.fit(X_train_new, y_train, validation_data=(X_val_new, y_val), batch_size = 100,epochs=EPOCH)
    histories_new.append(history)
    labels.append(i)
    export_training_accimage(history, i)
    pickle.dump(annmodel, open('Savedmodels_ANN/ANN_'+i+'.sav', 'wb'))
    #Results of Encodermodel


    val_acc = annmodel.predict(X_val_new)
    val_acc[val_acc>0.5]=1
    val_acc[val_acc<0.5]=0
    #print(accuracy_score(val_acc,y_val))
    accuracies.append(accuracy_score(val_acc,y_val))

    test_acc = annmodel.predict(X_test_new)
    test_acc[test_acc>0.5]=1
    test_acc[test_acc<0.5]=0
    #print(accuracy_score(test_acc,y_test))
    accuracies.append(accuracy_score(test_acc,y_test))

    for data in testfiles_new:
        accuracies.append(printacc(annmodel, data))

    print(classification_report(test_acc,y_test, target_names=['Regulation','No Regulation']))

    name_to_print = 'ANN_'+i

    if testfilenumber !=0:
        export_predictions_nn(annmodel, testfiles_new, name_to_print)

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
import datetime
import pytz
central = pytz.timezone('US/Eastern')
now = datetime.datetime.now(central)
date_string = now.strftime("%Y-%m-%d-%H-%M-%S")

accuracy_df.to_csv('Savedmodels_ANN/accuracy_table'+date_string+'.csv')
print('Accuracy values are saved in same folder and Predictions are saved at Finaldata/Predictions')

# motiffilenumber = int(input('Number of motif file available: '))
motiffilenumber = 0
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
 
