#Created by Sai Teja Mummadi, Computer Science, Michigan Technological University (Houghton, Mi)
#Used for the inference with transformation, 4D data, Requires Training data to transform the new test data.
#Input: training.csv, test1.csv, test2.csv
#Output: Accuracy_Table.csv, Predictions.csv

# Libraries
#Libraries for File operations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

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

#Dataset file must be in CSV format
#The ground truth column must be named as 'Regulation'
#The python file and the training and testing data must be in same folder

trainfilename = input('Enter the train file name Eg: Traindata.csv :')
train = pd.read_csv(trainfilename)

#Cleaning the training data
#Dropping the duplicate values, filling NULL values with 0
train = train.drop_duplicates()
train = train.reset_index(drop=True)
ytrain = train['Regulation']
train = train.drop(['Regulation'], axis=1)
train = train.fillna(0)


samplesize = int(input('Enter number of samples available for each TF: '))

#Processing Train data
train = train.iloc[:,2:]
traintf = train.iloc[:,:samplesize + 1]
traintarget = train.iloc[:,samplesize + 1:]
traintf = traintf.iloc[:,1:]
traintarget = traintarget.iloc[:,1:]
trainfull = traintf.join(traintarget)
trainfull = np.array(trainfull)
trainsc = MinMaxScaler()
trainsc.fit(trainfull)

traintransform = trainsc.transform(trainfull)
traintfnew = np.array(traintransform[:,0:samplesize])
traintargetnew = np.array(traintransform[:,samplesize:])
trainmultiply = traintfnew * traintargetnew
traindiff = np.subtract(traintfnew, traintargetnew)
trainfinal = []
for i, k in enumerate(traintfnew):
    trainfinal.append(np.vstack([(traintfnew[i],traintargetnew[i]),trainmultiply[i],traindiff[i]]).T)
trainfinal = np.array(trainfinal)
samplesizenew = samplesize - samplesize % 10
trainfinal = trainfinal[:,:samplesizenew,:]
trainfinal = trainfinal.reshape(trainfinal.shape[0],int(samplesizenew/10),10,4)

print(trainfinal.shape)
print(ytrain.shape)
# print(testfinal.shape)
os.makedirs('Finaldata', exist_ok=True)

np.save('Finaldata/Xtraindata',trainfinal)
np.save('Finaldata/ytraindata',ytrain)
print('Training data files are saved at Finaldata/Xtraindata.npy, Finaldata/ytraindata.npy')

os.makedirs('Finaldata/Predictions', exist_ok=True)

tfiles = []
tfilesns = []
            
#Cleaning the test data
number = int(input('Enter number of test files available: '))
for i in range(number):
    inputprompt = 'Enter the path of test file '+str(i+1)+' Eg: testdata.csv :'
    testfilename = input(inputprompt)
    test = pd.read_csv(testfilename)
    test = test.drop_duplicates()
    test = test.reset_index(drop=True)
    test = test.fillna(0)

    ##Processing Test data
    headers = test.iloc[:,:2]
    test = test.iloc[:,2:]
    testtf = test.iloc[:,:samplesize + 1]
    testtarget = test.iloc[:,samplesize + 1:]
    testtf = testtf.iloc[:,1:]
    testtarget = testtarget.iloc[:,1:]
    testfull = testtf.join(testtarget)
    testfull = np.array(testfull)
    testtransform = trainsc.transform(testfull)
    testtfnew = np.array(testtransform[:,0:samplesize])
    testtargetnew = np.array(testtransform[:,samplesize:])
    testmultiply = testtfnew * testtargetnew
    testdiff = np.subtract(testtfnew, testtargetnew)
    testfinal = []
    for i, k in enumerate(testtfnew):
        testfinal.append(np.vstack([(testtfnew[i],testtargetnew[i]),testmultiply[i],testdiff[i]]).T)    
    testfinal = np.array(testfinal)
    testfinal = testfinal[:,:samplesizenew,:]
    testfinal = testfinal.reshape(testfinal.shape[0],int(samplesizenew/10),10,4)
    testdataname = testfilename[:-4]
    headers.to_csv('Finaldata/Predictions/predvals'+testfilename, index=False)
    tfiles.append(testfinal)
    newtdname = testdataname+'.npy'
    tfilesns.append(newtdname)
    np.save('Finaldata/'+testdataname,testfinal)
    print('Test File is saved at location Finaldata/'+testdataname+'.npy')



testfilenumber = number
testfiles = tfiles
testfilenames = tfilesns
# for i in range(testfilenumber):
#     name  = input('Enter the test file '+str(i+1)+' name Eg:Testfilename.npy:')
#     testfilenames.append(name)
#     testfiles.append(np.load('Finaldata/'+name))
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

    cnnmodel=pickle.load(open('Savedmodels_cnn/CNN_'+i+'.sav', 'rb'))


    for data in testfiles:
        accuracies.append(printacc(cnnmodel, data))

    name_to_print = 'CNN_'+i

    if testfilenumber !=0:
        export_predictions_nn(cnnmodel, testfiles, name_to_print)

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
accuracy_df.to_csv('Savedmodels_cnn/accuracy_table'+date_string+'.csv')

#Merging with the motif files

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