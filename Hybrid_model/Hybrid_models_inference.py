#Created by Sai Teja Mummadi, Computer Science, Michigan Technological University (Houghton, Mi)
#Inference of Hybrid Models
#input: Xtrain.npy,ytrain.npy test.npy
#Output: Accuracy_Table.csv, Predictions.csv

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

#Accuracy function for test data
def printacc(model, data):
    predictions = model.predict(data)
    predictions = np.round(abs(predictions))
    positivelabels = (predictions==1).sum()
    negativelabels = (predictions==0).sum()
#    print('Negative:',negativelabels,'Positive:',positivelabels)
    rate = positivelabels/(negativelabels+positivelabels)
    return rate

##Accuracy Function for Validation and Testing data
def MLprediction(model, data):
    for file in data[0]:
        accuracies.append(printacc(model, file))
    #print(accuracies)

##Function to export the results to a CSV file
def export_predictions(model, data,testfilenames=testfilenames):
    for i,file in enumerate(data):
        testfilenam = testfilenames[i]
        outputfile = pd.read_csv('Finaldata/Predictions/predvals'+testfilenam[:-4]+'.csv')
        outputvalues = model.predict_proba(file)
        accuracylist = []
        for i in outputvalues:
            accuracylist.append(i[1])
        model_name = type(model).__name__
        outputfile[model_name] = accuracylist
        outputfile.to_csv('Finaldata/Predictions/predvals'+testfilenam[:-4]+'.csv',index=False)


def export_predictions_nn(model, data,testfilenames=testfilenames):
    for i,file in enumerate(data):
        testfilenam = testfilenames[i]
        outputfile = pd.read_csv('Finaldata/Predictions/predvals'+testfilenam[:-4]+'.csv')
        outputfile['Neuralnet'] = model.predict(file)
        outputfile.to_csv('Finaldata/Predictions/predvals'+testfilenam[:-4]+'.csv',index=False)

        
        
#Results of Encodermodel
accuracies = []
hybrid_model = pickle.load(open('Savedmodels/hybrid_model.sav', 'rb'))
for data in testfiles:
    accuracies.append(printacc(hybrid_model, data))
#print(accuracies)
encoder = pickle.load(open('Savedmodels/encoder.sav', 'rb'))
testfiles_new = []
for file in testfiles:
    testfiles_new.append(encoder.predict(file))

modelnames = []
modelnames.append('Encoder_Model')
data_new = [testfiles_new]


def NNmodelinference():
    annmodel = pickle.load(open('Savedmodels/neuralnet.sav', 'rb'))
    if testfilenumber !=0:
        export_predictions_nn(annmodel, testfiles_new)
    for data in testfiles_new:
        accuracies.append(printacc(annmodel, data))
    modelnames.append('ArtificialNeuralNetwork')

    
def Logisticmodelinference():
    print('Running Logistic Regression model')
    logistic_reg = pickle.load(open('Savedmodels/logistic_reg.sav', 'rb'))
    MLprediction(logistic_reg,data_new)
    if testfilenumber !=0:
        export_predictions(logistic_reg, testfiles_new)
    modelnames.append(type(logistic_reg).__name__)

def SVMmodelinference():
    print('Running Support Vector machine model ') 
    SVM_model = pickle.load(open('Savedmodels/SVM_model.sav', 'rb'))
    MLprediction(SVM_model,data_new)
    if testfilenumber !=0:
        export_predictions(SVM_model, testfiles_new)
    modelnames.append(type(SVM_model).__name__)

def DTEmodelinference():
    print('Running Decision Tree model')
    DTE_model = pickle.load(open('Savedmodels/DTE_model.sav', 'rb'))
    MLprediction(DTE_model,data_new)
    if testfilenumber !=0:
        export_predictions(DTE_model, testfiles_new)
    modelnames.append(type(DTE_model).__name__)

#K Nearest Neighbors
def KNNmodelinference():
    print('Running KNN model')
    KNN_model = pickle.load(open('Savedmodels/KNN_model.sav', 'rb'))
    MLprediction(KNN_model,data_new)
    modelnames.append(type(KNN_model).__name__)
    if testfilenumber !=0:
        export_predictions(KNN_model, testfiles_new)

#Random Forest
def RFmodelinference():
    print('Running Random Forest model') 
    randomforest = pickle.load(open('Savedmodels/randomforest.sav', 'rb'))
    MLprediction(randomforest,data_new)
    modelnames.append(type(randomforest).__name__)
    if testfilenumber !=0:
        export_predictions(randomforest, testfiles_new)


#Extra Tree Classifier
def ExtraTreemodelinference():
    print('Running Extra Tree classifier model')
    ETC_model = pickle.load(open('Savedmodels/ETC_model.sav', 'rb'))
    MLprediction(ETC_model,data_new)
    modelnames.append(type(ETC_model).__name__)
    if testfilenumber !=0:
        export_predictions(ETC_model, testfiles_new)


#Adaboost Classifier
def Adaboostmodelinference():
    print('Running Adaboost model')
    ADB_model = pickle.load(open('Savedmodels/ADB_model.sav', 'rb'))
    MLprediction(ADB_model,data_new)
    modelnames.append(type(ADB_model).__name__)
    if testfilenumber !=0:
        export_predictions(ADB_model, testfiles_new)


#Gradient Boosting Algorithm
def GradientBoostingmodelinference():
    print('Running Gradient Boosting Algorithm')
    GB_model = pickle.load(open('Savedmodels/GB_model.sav', 'rb'))
    MLprediction(GB_model,data_new)
    modelnames.append(type(GB_model).__name__)
    if testfilenumber !=0:
        export_predictions(GB_model, testfiles_new)

#Bagging Classifier
def baggingclassifierinference():
    print('Running Bagging classifier model')
#     BC_model = pickle.load(open('Savedmodels/BC_model.sav', 'rb'))
#     MLprediction(BC_model,data_new)
#     modelnames.append(type(BC_model).__name__)
#     if testfilenumber !=0:
#         export_predictions(BC_model, testfiles_new)
# Get the current date
central = pytz.timezone('US/Eastern')

now = datetime.datetime.now(central)

# Format the date as a string using strftime()
date_string = now.strftime("%Y-%m-%d-%H-%M-%S")

NNmodelinference()
Logisticmodelinference()
SVMmodelinference()
DTEmodelinference()
KNNmodelinference()
RFmodelinference()
ExtraTreemodelinference()
Adaboostmodelinference()
GradientBoostingmodelinference()
baggingclassifierinference()
accuracies2 = np.array(accuracies)
accuracies2 = accuracies2.reshape(len(modelnames),testfilenumber)
headernames = []
for i in range(testfilenumber):
    testfilenam = testfilenames[i]
    headernames.append(testfilenam[:-4])

accuracy_df = pd.DataFrame(accuracies2)
accuracy_df = accuracy_df.set_index([modelnames])
accuracy_df.columns = headernames    
accuracy_df.to_csv('Savedmodels/accuracy_table'+date_string+'.csv')

# motiffilenumber = int(input('Number of motif file available: '))
# motifcount = 0
# for i in range(testfilenumber):
#     if motifcount == motiffilenumber:
#         break
#     flag = input('Is motif file available for '+testfilenames[i]+'? Enter y or n:')
#     if(flag == 'y'):
#         motiffilename =  input('Enter the Motif file name or path: ')
#         motifdata = pd.read_csv(motiffilename)
#         newtestfilename = testfilenames[i]
#         preddata = pd.read_csv('Finaldata/Predictions/predvals'+newtestfilename[:-4]+'.csv')
#         result = pd.merge(preddata, motifdata,how="left",  on=["TF","Gene"])
#         result.to_csv('Finaldata/Predictions/predvals'+newtestfilename[:-4]+'.csv',index=False)
#         motifcount += 1
#     else:
#         pass
# print('The Final output files are saved at Finaldata/Predictions/') 

