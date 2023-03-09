#Created by Sai Teja Mummadi, Computer Science, Michigan Technological University (Houghton, Mi)
#Training and saving models hybrid models
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
appropriate_shape = trainshape[1:]

newpath = r'Savedmodels' 
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

#Different Loss Functions
los = tf.keras.losses.mean_squared_logarithmic_error
# los = tf.keras.losses.MeanSquaredError()
# los = 'binary_crossentropy'
# los = tf.keras.losses.MeanAbsoluteError()
# los = tf.keras.losses.Hinge()
# los = tf.keras.losses.Poisson()
# los = tf.keras.losses.Huber()
# los = tf.keras.losses.LogCosh()

#Change the number of Kernels
C1 = 32  #Other sizes could be 16, 32, 64, 128, 256 ...
C2 = 32  #Other sizes could be 16, 32, 64, 128, 256 ...
C3 = 64  #Other sizes could be 16, 32, 64, 128, 256 ...
C4 = 64  #Other sizes could be 16, 32, 64, 128, 256 ...

#Change Kernel size
ksize = 3 #Other size could be 3,4,5 and not more than 6

#Change encoder to add more CNN layers or Dense layers appropriately, 
#do not edit the first layer which takes input_shape.
encoder = Sequential()
encoder.add(Conv2D(C1, ksize, activation = activation, padding = 'same', input_shape = appropriate_shape))
encoder.add(BatchNormalization())
encoder.add(Conv2D(C2, ksize, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
encoder.add(BatchNormalization())
encoder.add(MaxPooling2D())
encoder.add(Conv2D(C3, ksize, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
encoder.add(BatchNormalization())
encoder.add(Conv2D(C4, ksize, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
encoder.add(BatchNormalization())
encoder.add(MaxPooling2D())
encoder.add(Flatten())
# encoder.add(Dense(1500, activation = activation, kernel_initializer = 'he_uniform'))
# encoder.add(Dense(1000, activation = activation, kernel_initializer = 'he_uniform'))


#Dense Network
#Change dense network appropriately to add more layers
layer = encoder.output
layer = Dense(128, activation = activation, kernel_initializer = 'he_uniform')(layer)
# layer = Dense(128, activation = activation, kernel_initializer = 'he_uniform')(layer)
final_layer = Dense(1, activation = 'sigmoid')(layer)

#Compiling the model
hybrid_model = Model(inputs=encoder.input, outputs=final_layer)
hybrid_model.compile(optimizer=opt,loss =los, metrics = ['accuracy'])
print(hybrid_model.summary())

#Training the Encoder model
history = hybrid_model.fit(X_train, y_train, epochs=EPOCH, validation_data = (X_val, y_val))

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
fig.suptitle('Accuracy and Loss')

# ax1.set(facecolor = "#D7DFF5")
# ax2.set(facecolor = "#D7DFF5")
ax1.set(xlabel='Number of Epochs', ylabel='Accuracy')
ax2.set(xlabel='Number of Epochs', ylabel='Loss')

ax1.legend(['Train accuracy', 'Validation accuracy'], loc='upper left')
ax2.legend(['Train Loss', 'Validation Loss'], loc='upper left')
ax1.grid()
ax2.grid()

os.makedirs('Figures', exist_ok=True)
plt.savefig('Figures/Encoder_training_history.png')
plt.close()

pickle.dump(encoder, open('Savedmodels/encoder.sav', 'wb'))
pickle.dump(hybrid_model, open('Savedmodels/hybrid_model.sav', 'wb'))

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

val_acc = hybrid_model.predict(X_val)
val_acc[val_acc>0.5]=1
val_acc[val_acc<0.5]=0
#print(accuracy_score(val_acc,y_val))
accuracies.append(accuracy_score(val_acc,y_val))

test_acc = hybrid_model.predict(X_test)
test_acc[test_acc>0.5]=1
test_acc[test_acc<0.5]=0
#print(accuracy_score(test_acc,y_test))
accuracies.append(accuracy_score(test_acc,y_test))


for data in testfiles:
    accuracies.append(printacc(hybrid_model, data))
#print(accuracies)

#Transforming Training data for ML models
X_train_new = encoder.predict(X_train)
X_val_new = encoder.predict(X_val)
X_test_new = encoder.predict(X_test)

testfiles_new = []
for file in testfiles:
    testfiles_new.append(encoder.predict(file))

data_new = [X_val_new, y_val, X_test_new, y_test,testfiles_new]
modelnames = []
modelnames.append('Encoder_Model')

###################################################
#Neural Network
def NNmodel():
    print('Running Neural Network model')
    inputlength = len(X_train_new[1])
    annmodel = Sequential()
    annmodel.add(Dense(2000, activation = activation,input_shape=(inputlength,), kernel_initializer = 'he_uniform'))
    annmodel.add(BatchNormalization())
    annmodel.add(Dense(1000, activation = activation, kernel_initializer = 'he_uniform'))
    annmodel.add(BatchNormalization())
    annmodel.add(Dense(500, activation = activation, kernel_initializer = 'he_uniform'))
    annmodel.add(BatchNormalization())
    annmodel.add(Dense(200, activation = activation, kernel_initializer = 'he_uniform'))
    annmodel.add(BatchNormalization())
    annmodel.add(Dense(1, activation = 'sigmoid'))
    annmodel.compile(optimizer=opt,loss =los, metrics = ['accuracy'])

    #Training the Encoder model
    annmodel.fit(X_train_new, y_train, epochs=80, validation_data = (X_val_new, y_val))


    val_acc_new = annmodel.predict(X_val_new)
    pickle.dump(annmodel, open('Savedmodels/neuralnet.sav', 'wb'))
    
    val_acc_new[val_acc_new>0.5]=1
    val_acc_new[val_acc_new<0.5]=0
    #print(accuracy_score(val_acc,y_val))
    accuracies.append(accuracy_score(val_acc_new,y_val))

    test_acc_new = annmodel.predict(X_test_new)
    test_acc_new[test_acc_new>0.5]=1
    test_acc_new[test_acc_new<0.5]=0
    #print(accuracy_score(test_acc,y_test))
    accuracies.append(accuracy_score(test_acc_new,y_test))
    print(classification_report(y_test, test_acc_new, target_names=['Regulation','No Regulation']))

    if testfilenumber !=0:
        export_predictions_nn(annmodel, testfiles_new)

    for data in testfiles_new:
        accuracies.append(printacc(annmodel, data))
    modelnames.append('ArtificialNeuralNetwork')




#Logistic Regression
def Logisticmodel():
    print('Running Logistic Regression model')
    logistic_reg = LogisticRegression(random_state=0,max_iter= 1000).fit(X_train_new, y_train)
    MLprediction(logistic_reg,data_new)
    if testfilenumber !=0:
        export_predictions(logistic_reg, testfiles_new)
    modelnames.append(type(logistic_reg).__name__)
    pickle.dump(logistic_reg, open('Savedmodels/logistic_reg.sav', 'wb'))

##SVM
def SVMmodel():
    print('Running Support Vector machine model ') 
    SVM_model = svm.SVC(kernel='rbf',random_state=123,probability=True)
    SVM_model.fit(X_train_new, y_train)
    MLprediction(SVM_model,data_new)
    if testfilenumber !=0:
        export_predictions(SVM_model, testfiles_new)
    modelnames.append(type(SVM_model).__name__)
    pickle.dump(SVM_model, open('Savedmodels/SVM_model.sav', 'wb'))
    
#Decision Tree with Entropy Criterion
def DTEmodel():
    print('Running Decision Tree model')
    DTE_model = DecisionTreeClassifier(criterion='entropy', random_state = 12)
    DTE_model.fit(X_train_new, y_train)    
    MLprediction(DTE_model,data_new)
    if testfilenumber !=0:
        export_predictions(DTE_model, testfiles_new)
    modelnames.append(type(DTE_model).__name__)
    pickle.dump(DTE_model, open('Savedmodels/DTE_model.sav', 'wb'))


#K Nearest Neighbors
def KNNmodel():
    print('Running KNN model')
    KNN_model = KNeighborsClassifier(n_neighbors=7)
    KNN_model.fit(X_train_new, y_train)    
    MLprediction(KNN_model,data_new)
    modelnames.append(type(KNN_model).__name__)
    if testfilenumber !=0:
        export_predictions(KNN_model, testfiles_new)
    pickle.dump(KNN_model, open('Savedmodels/KNN_model.sav', 'wb'))
#Random Forest
def RFmodel():
    print('Running Random Forest model') 
    randomforest = RandomForestClassifier(n_estimators = 500, random_state = 42)
    randomforest.fit(X_train_new, y_train)
    MLprediction(randomforest,data_new)
    modelnames.append(type(randomforest).__name__)
    if testfilenumber !=0:
        export_predictions(randomforest, testfiles_new)
    pickle.dump(randomforest, open('Savedmodels/randomforest.sav', 'wb'))

#Extra Tree Classifier
def ExtraTreemodel():
    print('Running Extra Tree classifier model')
    ETC_model = ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=10, random_state=0)
    ETC_model.fit(X_train_new, y_train)
    MLprediction(ETC_model,data_new)
    modelnames.append(type(ETC_model).__name__)
    if testfilenumber !=0:
        export_predictions(ETC_model, testfiles_new)
    pickle.dump(ETC_model, open('Savedmodels/ETC_model.sav', 'wb'))


#Adaboost Classifier
def Adaboostmodel():
    print('Running Adaboost model')
    ADB_model = AdaBoostClassifier(n_estimators=100)
    ADB_model.fit(X_train_new, y_train)
    MLprediction(ADB_model,data_new)
    modelnames.append(type(ADB_model).__name__)
    if testfilenumber !=0:
        export_predictions(ADB_model, testfiles_new)
    pickle.dump(ADB_model, open('Savedmodels/ADB_model.sav', 'wb'))

#Gradient Boosting Algorithm
def GradientBoostingmodel():
    print('Running Gradient Boosting Algorithm')
    GB_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    GB_model.fit(X_train_new, y_train)
    MLprediction(GB_model,data_new)
    modelnames.append(type(GB_model).__name__)
    if testfilenumber !=0:
        export_predictions(GB_model, testfiles_new)
    pickle.dump(GB_model, open('Savedmodels/GB_model.sav', 'wb'))

#Bagging Classifier
def baggingclassifier():
    print('Running Bagging classifier model')
    BC_model = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)
    BC_model.fit(X_train_new, y_train)
    MLprediction(BC_model,data_new)
    modelnames.append(type(BC_model).__name__)
    if testfilenumber !=0:
        export_predictions(BC_model, testfiles_new)
    pickle.dump(BC_model, open('Savedmodels/BC_model.sav', 'wb'))

while 1:    
    print('1: All models \n2: Logistic Regression \n3: Support Vector Machine \n4: Decision Tree \n5: K Nearest Neighbors \n6: Random Forest Model \n7: Extra Tree Classifier \n8: Adaboost Model \n9: Gradient Boosting Model \n10: Bagging Classifier \n11: Exit')
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
            NNmodel()
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
accuracy_df.to_csv('accuracy_table.csv')    
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

