# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:25:47 2019

@author: ABC
"""
import pandas as pd

from sklearn import  preprocessing



def read():

    # import apply_svm

#    dataset=pd.read_table('/home/shashank/Desktop/WDBC.dat',sep=',',header=None)

    

    

    dataset=pd.read_csv('data2023',sep=',',header=None)#读取以‘/t’分割的文件到dataframe

#    return dataset1

#x=read()

    #removing identifier
    #dataset = dataset1.sample(frac=1).reset_index(drop=True)#参数frac是要返回的比例，reset_index，可以还原索引，重新变为默认的整形所以
#return dataset

    #print(dataset)
#drop为true，把原本的索引index列去丢，丢掉
#    preprocessed_dataset=dataset.drop(0,1)
#    return preprocessed_dataset
#x=read()
#    print(preprocessed_dataset)

    

#    sh=preprocessed_dataset.shape

#    ind=[False]*sh[0]

#    for i in range(sh[0]):

#        for j in range(1,sh[1]):

#            if preprocessed_dataset.ix[i,j]=='?':

#                ind[i]=True

#    preprocessed_dataset.drop(preprocessed_dataset.index[ind],inplace=True)

#    preprocessed_dataset.reset_index(drop=True,inplace=True)

   

   # 2=benign,4= malignant

#    sh=preprocessed_dataset.shape

#    for i in range(sh[0]):

#        if preprocessed_dataset.ix[i,10]==2:

#            preprocessed_dataset.ix[i,10]='B'

#        else:

#            preprocessed_dataset.ix[i,10]='M'

#    

    features_no=len(dataset.columns)-1

    

    X=dataset.loc[:,1:]

    Y=dataset.loc[:,0]
    
    print(Y)

    

#    X=preprocessed_dataset.ix[:,1:9]    

#    Y=preprocessed_dataset.ix[:,10]

    

    #X_preprocessed=preprocessing.scale(X)

    #X_preprocessed=pd.DataFrame(X_preprocessed)
    #print(len(X_preprocessed[0]))

    return [X,Y,features_no]
#x=read()

    # apply_svm.apply(preprocessed_dataset.ix[:,2:31],preprocessed_dataset.ix[:,1],preprocessed_dataset.ix[119:120,2:31])
