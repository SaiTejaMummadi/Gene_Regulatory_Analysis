
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")



trainfilename = input('Enter the train file name Eg: Traindata.csv :')
train = pd.read_csv(trainfilename)
print(train.shape)
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

mask1 = (traintf == 0).all(axis=1)
idx_to_drop1 = mask1.index[mask1].tolist()
traintf.drop(idx_to_drop1, inplace=True)
traintarget.drop(idx_to_drop1, inplace=True)
ytrain.drop(idx_to_drop1, inplace=True)

mask2 = (traintarget == 0).all(axis=1)
idx_to_drop2 = mask2.index[mask2].tolist()
traintf.drop(idx_to_drop2, inplace=True)
traintarget.drop(idx_to_drop2, inplace=True)
ytrain.drop(idx_to_drop2, inplace=True)


trainfull = traintf.join(traintarget)
trainfull = np.array(trainfull)
trainsc = MinMaxScaler()
#trainsc = StandardScaler()
trainsc.fit(trainfull)

traintransform = trainsc.transform(trainfull)
samplesizenew = samplesize - samplesize % 10
trainfinal = traintransform[:,:samplesizenew*2]
trainfinal = trainfinal.reshape(trainfinal.shape[0],int(samplesizenew/10),20,1)

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
    print(test.shape)
    ##Processing Test data
    headers = test.iloc[:,:2]
    test = test.iloc[:,2:]
    testtf = test.iloc[:,:samplesize + 1]
    testtarget = test.iloc[:,samplesize + 1:]
    testtf = testtf.iloc[:,1:]
    testtarget = testtarget.iloc[:,1:]
    
    mask1 = (testtf == 0).all(axis=1)
    idx_to_drop1 = mask1.index[mask1].tolist()
    testtf.drop(idx_to_drop1, inplace=True)
    testtarget.drop(idx_to_drop1, inplace=True)
    headers.drop(idx_to_drop1, inplace=True)
    
    mask2 = (testtarget == 0).all(axis=1)
    idx_to_drop2 = mask2.index[mask2].tolist()
    testtf.drop(idx_to_drop2, inplace=True)
    testtarget.drop(idx_to_drop2, inplace=True)
    headers.drop(idx_to_drop2, inplace=True)    
    
    testfull = testtf.join(testtarget)
    testfull = np.array(testfull)
    testtransform = trainsc.transform(testfull)
    samplesizenew = samplesize - samplesize % 10
    testfinal = testtransform[:,:samplesizenew*2]
    testfinal = testfinal.reshape(testfinal.shape[0],int(samplesizenew/10),20,1)
    print(testfinal.shape)
    testdataname = testfilename[:-4]
    headers.to_csv('Finaldata/Predictions/predvals'+testfilename, index=False)
    np.save('Finaldata/'+testdataname,testfinal)
    print('Test File is saved at location Finaldata/'+testdataname+'.npy')
    
 
