#Created by Sai Teja Mummadi, Computer Science, Michigan Technological University (Houghton, Mi)
#Training and saving machine learning models 4D
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

newpath = r'Savedmodels_ml' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

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
    predval = model.predict(data[0])
    valacc_score = metrics.accuracy_score(data[1], predval)
    accuracies.append(valacc_score)

    predtest = model.predict(data[2])
    testacc_score = metrics.accuracy_score(data[3], predtest)
    accuracies.append(testacc_score)
    print(classification_report(data[3], predtest, target_names=['Regulation','No Regulation']))
    
    for file in data[4]:
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
        model_name = model_name + '_ml'
        outputfile[model_name] = accuracylist
        outputfile.to_csv('Finaldata/Predictions/predvals'+testfilenam[:-4]+'.csv',index=False)



        
#Results of Encodermodel
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

# Get the current date
central = pytz.timezone('US/Eastern')

now = datetime.datetime.now(central)

# Format the date as a string using strftime()
date_string = now.strftime("%Y-%m-%d-%H-%M-%S")


#Logistic Regression
def Logisticmodel():
    print('Running Logistic Regression model')
    logistic_reg = LogisticRegression(random_state=0,C=1,penalty='l2',solver='lbfgs',max_iter= 1000).fit(X_train_new, y_train)
    MLprediction(logistic_reg,data_new)
    if testfilenumber !=0:
        export_predictions(logistic_reg, testfiles_new)
    modelnames.append(type(logistic_reg).__name__)
    pickle.dump(logistic_reg, open('Savedmodels_ml/logistic_reg.sav', 'wb'))

##SVM
def SVMmodel():
    print('Running Support Vector machine model ') 
    SVM_model = svm.SVC(kernel='linear',C=0.5,degree=4, gamma='scale',probability=True,random_state=123)
    SVM_model.fit(X_train_new, y_train)
    MLprediction(SVM_model,data_new)
    if testfilenumber !=0:
        export_predictions(SVM_model, testfiles_new)
    modelnames.append(type(SVM_model).__name__)
    pickle.dump(SVM_model, open('Savedmodels_ml/SVM_model.sav', 'wb'))
    
#Decision Tree with Entropy Criterion
def DTEmodel():
    print('Running Decision Tree model')
    DTE_model = DecisionTreeClassifier(criterion='gini',min_samples_leaf=5,min_samples_split=5, random_state = 12)
    DTE_model.fit(X_train_new, y_train)    
    MLprediction(DTE_model,data_new)
    if testfilenumber !=0:
        export_predictions(DTE_model, testfiles_new)
    modelnames.append(type(DTE_model).__name__)
    pickle.dump(DTE_model, open('Savedmodels_ml/DTE_model.sav', 'wb'))


#K Nearest Neighbors
def KNNmodel():
    print('Running KNN model')
    KNN_model = KNeighborsClassifier(n_neighbors=7,metric='manhattan',weights='distance')
    KNN_model.fit(X_train_new, y_train)    
    MLprediction(KNN_model,data_new)
    modelnames.append(type(KNN_model).__name__)
    if testfilenumber !=0:
        export_predictions(KNN_model, testfiles_new)
    pickle.dump(KNN_model, open('Savedmodels_ml/KNN_model.sav', 'wb'))
#Random Forest
def RFmodel():
    print('Running Random Forest model') 
    randomforest = RandomForestClassifier(n_estimators = 50,criterion='log_loss',max_depth=20,min_samples_leaf=5,
                                          min_samples_split=5, random_state = 42)
    randomforest.fit(X_train_new, y_train)
    MLprediction(randomforest,data_new)
    modelnames.append(type(randomforest).__name__)
    if testfilenumber !=0:
        export_predictions(randomforest, testfiles_new)
    pickle.dump(randomforest, open('Savedmodels_ml/randomforest.sav', 'wb'))

#Extra Tree Classifier
def ExtraTreemodel():
    print('Running Extra Tree classifier model')
    ETC_model = ExtraTreesClassifier(n_estimators=50, criterion='log_loss',
                                     max_depth=20, min_samples_split=5,min_samples_leaf=1, random_state=0)
    ETC_model.fit(X_train_new, y_train)
    MLprediction(ETC_model,data_new)
    modelnames.append(type(ETC_model).__name__)
    if testfilenumber !=0:
        export_predictions(ETC_model, testfiles_new)
    pickle.dump(ETC_model, open('Savedmodels_ml/ETC_model.sav', 'wb'))


#Adaboost Classifier
def Adaboostmodel():
    print('Running Adaboost model')
    ADB_model = AdaBoostClassifier(n_estimators=100)
    ADB_model.fit(X_train_new, y_train)
    MLprediction(ADB_model,data_new)
    modelnames.append(type(ADB_model).__name__)
    if testfilenumber !=0:
        export_predictions(ADB_model, testfiles_new)
    pickle.dump(ADB_model, open('Savedmodels_ml/ADB_model.sav', 'wb'))

#Gradient Boosting Algorithm
def GradientBoostingmodel():
    print('Running Gradient Boosting Algorithm')
    GB_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    GB_model.fit(X_train_new, y_train)
    MLprediction(GB_model,data_new)
    modelnames.append(type(GB_model).__name__)
    if testfilenumber !=0:
        export_predictions(GB_model, testfiles_new)
    pickle.dump(GB_model, open('Savedmodels_ml/GB_model.sav', 'wb'))

#Bagging Classifier
def baggingclassifier():
    print('Running Bagging classifier model')
    BC_model = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)
    BC_model.fit(X_train_new, y_train)
    MLprediction(BC_model,data_new)
    modelnames.append(type(BC_model).__name__)
    if testfilenumber !=0:
        export_predictions(BC_model, testfiles_new)
    pickle.dump(BC_model, open('Savedmodels_ml/BC_model.sav', 'wb'))

while 1:    
    print('1: All models  \n2: Logistic Regression \n3: Support Vector Machine \n4: Decision Tree \n5: K Nearest Neighbors \n6: Random Forest Model \n7: Extra Tree Classifier \n8: Adaboost Model \n9: Gradient Boosting Model \n10: Bagging Classifier \n11: Exit')
    modelnum = int(input('Enter a model number to Run:'))
    if(modelnum == 2):
        Logisticmodel()
    elif(modelnum == 3):
            SVMmodel()
    elif(modelnum == 4):
            DTEmodel()
    elif(modelnum == 5):
            KNNmodel()
    elif(modelnum == 6):
            RFmodel()
    elif(modelnum == 7):
            ExtraTreemodel()
    elif(modelnum == 8):
            Adaboostmodel()
    elif(modelnum == 9):
            GradientBoostingmodel()
    elif(modelnum == 10):
            baggingclassifier()
    elif(modelnum == 1):
            Logisticmodel()
            SVMmodel()
            DTEmodel()
            KNNmodel()
            RFmodel()
            ExtraTreemodel()
            Adaboostmodel()
            GradientBoostingmodel()
            baggingclassifier()
    elif(modelnum == 11):
        break
    else:
        print('Enter a number between 1 and 11: ')


accuracies2 = np.array(accuracies)
accuracies2 = accuracies2.reshape(len(modelnames),testfilenumber+2)
headernames = ['Validation','Testing']
for i in range(testfilenumber):
    testfilenam = testfilenames[i]
    headernames.append(testfilenam[:-4])

accuracy_df = pd.DataFrame(accuracies2)
accuracy_df = accuracy_df.set_index([modelnames])
accuracy_df.columns = headernames    
accuracy_df.to_csv('Savedmodels_ml/accuracy_table'+date_string+'.csv')
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
