# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:31:36 2019

@author: ABC
"""
#from python_speech_features import mfcc
from scipy.fftpack import dct
import numpy as np

from scipy.io import wavfile

#from gfcc1 import cochleagram_extractor

from matplotlib import  pyplot as plt

#from speech_utils import read_sphere_wav
#import librosa
import time
#import librosa
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score#normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数
#import pandas as pd 
#import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn
#from keras.models import load_model
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import precision_score
import datetime as dt
#from sklearn import svm
#from sklearn.datasets import load_iris
#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
#import numpy as np
import os, pickle, random, datetime
#from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
import read_data
import sklearn.metrics
[X,Y_,features_no]=read_data.read('data2023')
sonar2=np.mat(X)
'''Y=Y_.reshape((2000,1))
data=np.hstack((Y,X))
np.savetxt('data331', data, delimiter=',')'''
#print(Y)
#x_train=sonar2
#y_train=Y_
#[T,V,FEATURE]=read_data.read('data42')
#sonar3=np.mat(T)
#V_=V
#x_test=sonar3
#y_test=V_
x_train=sonar2[0:1000]
y_train=Y_[0:1000]
x_test=sonar2[1000:,:]
y_test=Y_[1000:]
'''filename = '299.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
pos = []
Efield = []
with open(filename, 'r') as file_to_read:
  while True:
    lines = file_to_read.readline() # 整行读取数据
    if not lines:
        break
        pass
    p_tmp= [float(i) for i in lines.split(',')] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
    pos.append(p_tmp)  # 添加新读取的数据
    #Efield.append(E_tmp)
    pass
pos = np.array(pos).T # 将数据从list类型转换为array类型。
pass'''
'''sonar3=np.zeros((2000,1))
s=0
for i in range(398):
    t=pos[i]
    if t==True:
        q=sonar2[:,i]
        q=q.reshape(2000,1)
        sonar3=np.append(sonar3,q,axis=1)
sonar3 = np.append(sonar3,q,axis=1)'''
'''Feature = np.zeros(299)        #数组Feature用来存 x选择的是哪d个特征 
k = 0
for i in range(398):
    if pos[i] == 1:
        Feature[k] = i
        k+=1
sonar3 = np.zeros((2000,1))
for i in range(299):
    p = Feature[i]
    p = p.astype(int)
    q = sonar2[:,p]
    q = q.reshape(2000,1)
    sonar3 = np.append(sonar3,q,axis=1)
sonar3 = np.delete(sonar3,0,axis=1)
x_train=sonar3[0:1000,:]
y_train=Y_[0:1000]
x_test=sonar3[1000:,:]
y_test=Y_[1000:]'''
'''x_test=np.load('test_x01.npy')
y_test=np.load('test_y01.npy')
x_train=np.load('train_x2020.npy')
y_train=np.load('train_y2020.npy')'''
param_grid = {'C': [10],'gamma': [0.01]}
#param_grid = {'C': [1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4,1e5],'gamma': [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)   #网格搜索+交叉验证
grid_search.fit(x_train,y_train)
end_time = dt.datetime.now() #结束训练的时间
joblib.dump(grid_search, "4_18_train_model.m")#保存模型

print('网格搜索-度量记录：',grid_search.cv_results_)  # 包含每次训练的相关信息
print('网格搜索-最佳度量值:',grid_search.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：',grid_search.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳模型：',grid_search.best_estimator_)  # 获取最佳度量时的分类器模型

print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search.best_params_)
print(' ')
print(' ')
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search.best_score_)#?????   
# print(' ')
print('BEST_ESTIMATOR:',grid_search.best_estimator_)   #对应分数最高的估计器
print(' ')
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在测试集上的得分',grid_search.score(x_test, y_test))#####

means = grid_search.cv_results_['mean_test_score']#具体的参数间不同数值的组合后得到的分数是多少：
stds = grid_search.cv_results_['std_test_score']

# 看一下具体的参数间不同数值的组合后得到的分数是多少
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

    #elapsed_time1= end_time - start_time1#整个训练时间包括数据处理时间
    #print('the training time Elapsed learning1 {}'.format(str(elapsed_time1)))

# Now predict the value of the test
expected = y_test#测试标签
predicted = grid_search.predict(x_test)#预测标签

print("Classification report for classifier %s:\n%s\n"
          % (grid_search,classification_report(expected, predicted)))
      
cm = confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

train_hat = grid_search.predict(x_train)
print("Train Accuracy={}".format(accuracy_score(y_train, train_hat)))#训练数据的准确率
test_hat = grid_search.predict(x_test)
print("Test Accuracy={}".format(accuracy_score(y_test, test_hat)))#测试数据的准确率

    #time_end= time.time() - time_start
   # print('Training complete in {:.0f}m {:.0f}s'.format(
       #     time_end // 60, time_end % 60)) # 打印出来时间 
print('10db')
    
    
print('pre',sklearn.metrics.precision_score(y_test, test_hat, average='binary'))
print(sklearn.metrics.recall_score(y_test, test_hat, average='binary'))
print(sklearn.metrics.f1_score(y_test, test_hat, average='binary'))

'''x_test=np.load('test_x2028.npy')
y_test=np.load('test_y2028.npy')
x_train=np.load('train_x2020.npy')
y_train=np.load('train_y2020.npy')'''
'''start_time1 = dt.datetime.now()#程序开始运行的时间
time_start=time.time()
num_class=2
start_time2=dt.datetime.now()
    
print('Start learning at {}'.format(str(start_time2)))
#param_grid = {'C': [2e-5,2e-4,2e-3,2e-2,2e-1,2e-0,2e1,2e2,2e3,2e4,2e5],'gamma': [2e-5,2e-4,2e-3,2e-2,2e-1,2e0,2e1,2e2,2e3,2e4,2e5]}
#param_grid = {'C': [5e-5,5e-4,5e-3,5e-2,5e-1,5e-0,5e1,5e2,5e3,5e4,5e5,5e6,5e7],'gamma': [5e-5,5e-4,5e-3,5e-2,5e-1,5e0,5e1,5e2,5e3,5e4,5e5,5e6,5e7]}
#param_grid = {'C': [4e-5,4e-4,4e-3,4e-2,4e-1,4e-0,4e1,4e2,4e3,4e4,4e5,4e6,4e7],'gamma': [4e-5,4e-4,4e-3,4e-2,4e-1,4e-0,4e1,4e2,4e3,4e4,4e5,4e6,4e7]}
#param_grid = {'C': [10],'gamma': [0.01]}

#param_grid = {'C': [9e-5,9e-4,9e-3,9e-2,9e-1,9e-0,9e1,9e2,9e3,9e4,9e5,9e6,9e7],'gamma': [9e-5,9e-4,9e-3,9e-2,9e-1,9e0,9e1,9e2,9e3,9e4,9e5,9e6,9e7]}

param_grid = {'C': [1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4],'gamma': [1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]}
#param_grid = {'C': [3e-5,3e-4,3e-3,3e-2,3e-1,3e0,3e1,3e2,3e3,3e4,3e5,3e6,3e7],'gamma': [3e-5,3e-4,3e-3,3e-2,3e-1,3e0,3e1,3e2,3e3,3e4,3e5,3e6,3e7]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)   #网格搜索+交叉验证
grid_search.fit(x_train,y_train)
end_time = dt.datetime.now() #结束训练的时间
joblib.dump(grid_search, "4_18_train_model.m")#保存模型

print('网格搜索-度量记录：',grid_search.cv_results_)  # 包含每次训练的相关信息
print('网格搜索-最佳度量值:',grid_search.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：',grid_search.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('网格搜索-最佳模型：',grid_search.best_estimator_)  # 获取最佳度量时的分类器模型

print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search.best_params_)
print(' ')
print(' ')
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search.best_score_)#?????   
# print(' ')
print('BEST_ESTIMATOR:',grid_search.best_estimator_)   #对应分数最高的估计器
print(' ')
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在测试集上的得分',grid_search.score(x_test, y_test))#####

means = grid_search.cv_results_['mean_test_score']#具体的参数间不同数值的组合后得到的分数是多少：
stds = grid_search.cv_results_['std_test_score']

# 看一下具体的参数间不同数值的组合后得到的分数是多少
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

elapsed_time1= end_time - start_time1#整个训练时间包括数据处理时间
print('the training time Elapsed learning1 {}'.format(str(elapsed_time1)))

# Now predict the value of the test
expected = y_test#测试标签
predicted = grid_search.predict(x_test)#预测标签

print("Classification report for classifier %s:\n%s\n"
          % (grid_search,classification_report(expected, predicted)))
      
cm = confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

train_hat = grid_search.predict(x_train)
print("Train Accuracy={:.6f}".format(accuracy_score(y_train, train_hat)))#训练数据的准确率
test_hat = grid_search.predict(x_test)
print("Test Accuracy={:.6f}".format(accuracy_score(y_test, test_hat)))#测试数据的准确率

time_end= time.time() - time_start
print('Training complete in {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60)) # 打印出来时间 '''

'''print(sklearn.metrics.precision_score(y_test, test_hat, average='binary'))
print(sklearn.metrics.recall_score(y_test, test_hat, average='binary'))
print(sklearn.metrics.f1_score(y_test, test_hat, average='binary'))'''



