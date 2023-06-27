#ANN model inference

## Libraries
#Libraries for File operations
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

testfilenumber = int(input('Enter the number of test files: '))
testfiles = []
testfilenames = []
for i in range(testfilenumber):
    name  = input('Enter the test file '+str(i+1)+' name Eg:Testfilename.npy:')
    testfilenames.append(name)
    testfiles.append(np.load('Finaldata/'+name))

    
    
testfiles_new = []
for file in testfiles:
    testfiles_new.append(np.reshape(file, (file.shape[0], -1)))
# data_new = [testfiles_new]

modelnames = []

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
        
#Results of Encodermodel
accuracies = []
modelnames = []


for i in losdict:

    annmodel=pickle.load(open('Savedmodels_ANN/ANN_'+i+'.sav', 'rb'))
    print('Running '+i+' Model')

    for data in testfiles_new:
        accuracies.append(printacc(annmodel, data))

    name_to_print = 'ANN_'+i

    if testfilenumber !=0:
        export_predictions_nn(annmodel, testfiles_new, name_to_print)
    
    modelnames.append(name_to_print)

accuracies2 = np.array(accuracies)
accuracies2 = accuracies2.reshape(len(modelnames),testfilenumber)
headernames = []
for i in range(testfilenumber):
    testfilenam = testfilenames[i]
    headernames.append(testfilenam[:-4])

# Get the current date
central = pytz.timezone('US/Eastern')

now = datetime.datetime.now(central)

# Format the date as a string using strftime()
date_string = now.strftime("%Y-%m-%d-%H-%M-%S")


accuracy_df = pd.DataFrame(accuracies2)
accuracy_df = accuracy_df.set_index([modelnames])
accuracy_df.columns = headernames    
accuracy_df.to_csv('Savedmodels_ANN/accuracy_table'+date_string+'.csv')
print('Results are saved')

# Merging with the motif files

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