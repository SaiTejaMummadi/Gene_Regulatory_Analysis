#Created by: Sai Teja Mummadi, Computer Science, Michigan Technological University(MI)
#Input: train.csv, test.csv
#Output: Files saved in .npy
#
#
#Execution Instructions:
#Dataset file must be in CSV format
#The ground truth column must be named as 'Regulation'
#The python file and the training and testing data must be in same folder


## Libraries
#Libraries for File operations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")



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
    np.save('Finaldata/'+testdataname,testfinal)
    print('Test File is saved at location Finaldata/'+testdataname+'.npy')
    
 