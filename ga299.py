# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:38:39 2019

@author: 11
"""

# -*- coding: utf-8 -*-

"""

Created on Sat Nov 10 17:51:47 2018



@author: 31723

"""
from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import datetime
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import read_data
import heapq
[X,Y,features_no]=read_data.read()

#sonar = pd.read_csv('sonar.txt',header=None,sep=',')

#sonar1 = sonar.iloc[0:208,0:60]

#sonar2 = np.mat(sonar1) 
sonar2=np.mat(X)
Y_=Y




pc = 0.1                            # pc为变异的概率

t = 200   #遗传算法迭代的次数

n = 30    #种群的个体数,要求大于20以保证具有随机性

d =299   # d为要选择的特征个数



#遗传算法

def GA(d):

    population = np.zeros((n,398))     # 初始化种群
    

    for i in range(n):                # 定义种群的个体数为 n

        a = np.zeros(398-d)

        b = np.ones(d)                # 将选择的d维特征定义为个体c中的1

        c = np.append(a,b)
        #print(c)

        c = (np.random.permutation(c.T)).T # 随机生成一个d维的个体

        population[i] = c             # 初代的种群为 population，共有n个个体

        

    #遗传算法的迭代次数为t

    fitness_change = np.zeros(t)
    fitness = np.zeros(n)             # fitness为每一个个体的适应度值
    for j in range(n):
        print(j)
        fitness[j] = Jd(population[j])# 计算每一个体的适应度值
        with open('299_clean.txt','a') as file:
            l=str(fitness[j])
            file.write('\n')
            file.write(l)


    for i in range(t):
        fitness1 = np.zeros(n)

 
        with open('299_clean.txt','a') as file:
            l=str(i)
            file.write('\n')
            file.write(l)
            
        
        '''re1 = heapq.nlargest(3, fitness) #求最大的三个元素，并排序
        re2 = map(fitness.index, heapq.nlargest(3, fitness)) #求最大的三个索引    nsmallest与nlargest相反，求最小
        re2=list(re2)
        fitness=list(fitness)
        f1=fitness[re2[0]]
        f2=fitness[re2[1]]
        f3=fitness[re2[2]]
        population=list(population)
        p1=population[re2[0]]
        p2=population[re2[1]]
        p3=population[re2[2]]
        del(population(re2[0]))
        del(population(re2[1]))
        del(population(re2[2]))
        population=np.array(population)
        del(fitness[re2[0]])
        del(fitness[re2[1]])
        del(fitness[re2[2]])
        fitness=np.array(fitness)'''
        #new_pop=population
        #new_pop=list(new_pop)
        #new_pop.append(a)
        #new_pop.append(b)
        #new_pop.append(c)
        #new_pop_last=np.array(new_pop)
        
        population2 = selection(population,fitness)  # 通过概率选择产生新一代的种群

        population2 = crossover(population2)          # 通过交叉产生新的个体

        population2 = mutation(population2)           # 通过变异产生新个体
        for m in range(n):
            fitness1[m]=Jd(population2[m])
        fitness1=list(fitness1)
        fitness=list(fitness)
        #fitness fot 2
        population=list(population)
        population2=list(population2)
        for i in range(n):
            population.append(population2[i])#将两个种群合并成一个种群
            fitness.append(fitness1[i])#将两个种群的适应度值合并成一个适应度列表
        #re1 = heapq.nlargest(30, fitness) #求最大的30个元素，并排序
        
        re2 =list(map(fitness.index, heapq.nlargest(30, fitness))) #求最大的30个索引
        fitness3=np.zeros(n)
        fitness3=list(fitness3)
        population3=np.zeros((n,398))
        #population3=list(population3)
        k=0
        for s in re2:
            population3[k]=population[s]#将适应度值最大的30个个体保留下来
            fitness3[k]=fitness[s]#将适应度值最大的30个个体的适应度值保留下来
            k=k+1
        population=np.array(population3)
        fitness=np.array(fitness3)
        for i in range(n):
            with open('299_clean.txt','a') as file:
                file.write('\n')
                m=str(fitness[i])
                #file.write('\n')
                file.write(m)
        
            
        
        
        
        
        '''population=list(population)
        population.append(p1)
        population.append(p2)
        population.append(p3)
        population=np.array(population)
        fitness=list(fitness)
        fitness.append(f1)
        fitness.append(f2)
        fitness.append(f3)'''
        

#        fitness_change[i] = max(fitness)
        
            

        '''population = selection(population,fitness)  # 通过概率选择产生新一代的种群

        population = crossover(population)          # 通过交叉产生新的个体

        population = mutation(population)           # 通过变异产生新个体

        fitness_change[i] = max(fitness) '''     #找出每一代的适应度最大的染色体的适应度值

        

        

    # 随着迭代的进行，每个个体的适应度值应该会不断增加，所以总的适应度值fitness求平均应该会变大

    best_fitness = max(fitness)
    with open('299_clean.txt','a') as file:
        file.write('\n')
        file.write('best_fitness')
        file.write('\n')
        m=str(best_fitness)
        #file.write('\n')
        file.write(m)

    best_people = population[fitness.argmax()]

    

    return best_people,best_fitness,fitness_change,population

    

    





#轮盘赌选择

def selection(population,fitness):

    fitness_sum = np.zeros(n)

    for i in range(n):

        if i==0:

            fitness_sum[i] = fitness[i]

        else:

            fitness_sum[i] = fitness[i] + fitness_sum[i-1]

    for i in range(n):

        fitness_sum[i] = fitness_sum[i] / sum(fitness)

    

    #选择新的种群

    population_new = np.zeros((n,398))

    for i in range(n):

        rand = np.random.uniform(0,1)

        for j in range(n):

            if j==0:

                if rand<=fitness_sum[j]:

                    population_new[i] = population[j]

            else:

                if fitness_sum[j-1]<rand and rand<=fitness_sum[j]:

                    population_new[i] = population[j]

    return population_new

                



#交叉操作

def crossover(population):

    father = population[0:10,:]

    mother = population[10:,:]

    np.random.shuffle(father)       # 将父代个体按行打乱以随机配对

    np.random.shuffle(mother)

    for i in range(10):

        father_1 = father[i]

        mother_1 = mother[i]

        one_zero = []

        zero_one = []

        for j in range(398):

            if father_1[j]==1 and mother_1[j]==0:

                one_zero.append(j)

            if father_1[j]==0 and mother_1[j]==1:

                zero_one.append(j)

        length1 = len(one_zero)

        length2 = len(zero_one)

        length = max(length1,length2)

        half_length = int(length/2)        #half_length为交叉的位数 

        for k in range(half_length):       #进行交叉操作

            p = one_zero[k]

            q = zero_one[k]

            father_1[p]=0

            mother_1[p]=1

            father_1[q]=1

            mother_1[q]=0

        father[i] = father_1               #将交叉后的个体替换原来的个体

        mother[i] = mother_1

    population = np.append(father,mother,axis=0)

    return population

                

            

    

#变异操作

def mutation(population):

    for i in range(n):

        c = np.random.uniform(0,1)

        if c<=pc:

            mutation_s = population[i]

            zero = []                           # zero存的是变异个体中第几个数为0

            one = []                            # one存的是变异个体中第几个数为1

            for j in range(398):

                if mutation_s[j]==0:

                    zero.append(j)

                else:

                    one.append(j)

                    

            if (len(zero)!=0) and (len(one)!=0):

                a = np.random.randint(0,len(zero))    # e是随机选择由0变为1的位置

                b = np.random.randint(0,len(one))     # f是随机选择由1变为0的位置

                e = zero[a]

                f = one[b]

                mutation_s[e] = 1

                mutation_s[f] = 0

                population[i] = mutation_s

            

    return population





#个体适应度函数 Jd(x)，x是d维特征向量(1*60维的行向量,1表示选择该特征)
def Jd(x):
    Feature = np.zeros(d)        #数组Feature用来存 x选择的是哪d个特征 

    k = 0

    for i in range(398):

        if x[i] == 1:

            Feature[k] = i

            k+=1

    

    #将30个特征从sonar2数据集中取出重组成一个208*d的矩阵sonar3

    sonar3 = np.zeros((2000,1))

    for i in range(d):

        p = Feature[i]

        p = p.astype(int)

        q = sonar2[:,p]

        q = q.reshape(2000,1)

        sonar3 = np.append(sonar3,q,axis=1)
        #print(sonar3)

    sonar3 = np.delete(sonar3,0,axis=1)
    #print(sonar3)
    #求准确率
    x_train=sonar3[0:1000,:]
    y_train=Y_[0:1000]
    x_test=sonar3[1000:,:]
    y_test=Y_[1000:]
    '''model = svm.SVC(C=100,gamma=0.001)
    model.fit(x_train, y_train)
    #print(model.score(X_train, Y_train))
#Predict Output
    predicted= model.predict(x_train)
    print("Train Accuracy={}".format(accuracy_score(y_train, predicted)))
    test_hat = model.predict(x_test)
    t=accuracy_score(y_test, test_hat)
    print("Test Accuracy={}".format(t))'''
    #param_grid = {'C': [1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4,1e5,1e6,1e7],'gamma': [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7]}
    #time_start=time.time()
    #param_grid = {'C': [1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4,1e5,1e6,1e7],'gamma': [1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4,1e5,1e6,1e7]}
    #param_grid = {'C': [2e-5,2e-4,2e-3,2e-2,2e-1,2e-0,2e1,2e2,2e3,2e4,2e5,2e6,2e7],'gamma': [2e-5,2e-4,2e-3,2e-2,2e-1,2e-0,2e1,2e2,2e3,2e4,2e5,2e6,2e7]}
    param_grid = {'C': [10],'gamma': [0.01]}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)   #网格搜索+交叉验证
    grid_search.fit(x_train,y_train)
#end_time = dt.datetime.now() #结束训练的时间
    joblib.dump(grid_search, "4_18_train_model.m")#保存模型

#print('网格搜索-度量记录：',grid_search.cv_results_)  # 包含每次训练的相关信息
#print('网格搜索-最佳度量值:',grid_search.best_score_)  # 获取最佳度量值
#print('网格搜索-最佳参数：',grid_search.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
#print('网格搜索-最佳模型：',grid_search.best_estimator_)  # 获取最佳度量时的分类器模型

#print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search.best_params_)
#print(' ')
#print(' ')
#print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search.best_score_)#?????   
# print(' ')
#print('BEST_ESTIMATOR:',grid_search.best_estimator_)   #对应分数最高的估计器
#print(' ')
#print('GridSearchCV交叉验证网格搜索获得的最好估计器,在测试集上的得分',grid_search.score(x_test, y_test))#####

    #means = grid_search.cv_results_['mean_test_score']#具体的参数间不同数值的组合后得到的分数是多少：
    #stds = grid_search.cv_results_['std_test_score']

# 看一下具体的参数间不同数值的组合后得到的分数是多少
    #for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        #print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

#elapsed_time1= end_time - start_time1#整个训练时间包括数据处理时间
#print('the training time Elapsed learning1 {}'.format(str(elapsed_time1)))

# Now predict the value of the test
    #expected = y_test#测试标签
    #predicted = grid_search.predict(x_test)#预测标签

    #print("Classification report for classifier %s:\n%s\n"% (grid_search,classification_report(expected, predicted)))
    #m=classification_report(expected, predicted)
#cm = confusion_matrix(expected, predicted)
#print("Confusion matrix:\n%s" % cm)

    train_hat = grid_search.predict(x_train)
    print("Train Accuracy={}".format(accuracy_score(y_train, train_hat)))#训练数据的准确率
    test_hat = grid_search.predict(x_test)
    
    print("Test Accuracy={}".format(accuracy_score(y_test, test_hat)))#测试数据的准确率
    t=accuracy_score(y_test, test_hat)

    #time_end= time.time() - time_start
    #print(time_end)
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60)) # 打印出来时间 

    return t








'''def Jd(x):

    #从特征向量x中提取出相应的特征

    Feature = np.zeros(d)        #数组Feature用来存 x选择的是哪d个特征 

    k = 0

    for i in range(60):

        if x[i] == 1:

            Feature[k] = i

            k+=1

    

    #将30个特征从sonar2数据集中取出重组成一个208*d的矩阵sonar3

    sonar3 = np.zeros((208,1))

    for i in range(d):

        p = Feature[i]

        p = p.astype(int)

        q = sonar2[:,p]

        q = q.reshape(208,1)

        sonar3 = np.append(sonar3,q,axis=1)

    sonar3 = np.delete(sonar3,0,axis=1)

    

    #求类间离散度矩阵Sb

    sonar3_1 = sonar3[0:97,:]        #sonar数据集分为两类

    sonar3_2 = sonar3[97:208,:]

    m = np.mean(sonar3,axis=0)       #总体均值向量

    m1 = np.mean(sonar3_1,axis=0)    #第一类的均值向量

    m2 = np.mean(sonar3_2,axis=0)    #第二类的均值向量

    m = m.reshape(d,1)               #将均值向量转换为列向量以便于计算

    m1 = m1.reshape(d,1)

    m2 = m2.reshape(d,1)

    Sb = ((m1 - m).dot((m1 - m).T)*(97/208) + (m2 - m).dot((m2 - m).T)*(111/208)) #除以类别个数

  

    #求类内离散度矩阵Sw

    S1 = np.zeros((d,d))

    S2 = np.zeros((d,d))

    for i in range(97):

        S1 += (sonar3_1[i].reshape(d,1)-m1).dot((sonar3_1[i].reshape(d,1)-m1).T)

    S1 = S1/97

    for i in range(111):

        S2 += (sonar3_2[i].reshape(d,1)-m2).dot((sonar3_2[i].reshape(d,1)-m2).T)

    S2 = S2/111

    

    Sw = (S1*(97/208) + S2*(111/208))

   # Sw = (S1 + S2) / 2

    #计算个体适应度函数 Jd(x)

    J1 = np.trace(Sb)

    J2 = np.trace(Sw)

    Jd = J1/J2

    

    return Jd'''

    





if __name__ == '__main__':

    #starttime1 = datetime.datetime.now()

    best_d = np.zeros(398)          # judge存的是每一个维数的最优适应度
    for j in range(30):
        for d in range(299,300):            # d为从60个特征中选择的特征维数
            best_people,best_fitness,fitness_change,best_population = GA(d)     # fitness_change是遗传算法在迭代过程中适应度变化

            best_d[d-1] = best_fitness     # best是每一维数迭代到最后的最优的适应度，用于比较

            print("在取%d维的时候，通过遗传算法得出的最优适应度值为：%.6f"%(d,best_fitness))
        with open('30lun_299clean_bestindividual.txt','a') as file:
            file.write(str(j))
            file.write('\n')
            u=str(list(best_people))
            file.write(u)
        with open('30lun_299clean.txt','a') as file:
            m=str(best_fitness)
            file.write('\n')
            file.write('第'+str(j)+'轮的最优适应度为：')
            file.write(m) 
    



    '''

    若要看遗传算法的收敛情况，则看在d=30的情况下的fitness_change就可以

    '''

    

    

    '''

    #画图

    x = np.arange(0,59,1)

    plt.xlabel('dimension')

    plt.ylabel('fitness')

    plt.ylim((0,0.3))            # y坐标的范围

    plt.plot(x,best_d,'r')

    # plt.savefig("Sonar_best_d.jpg",dpi=2000)

    '''

    

    #endtime1 = datetime.datetime.now()
    #time1 = endtime1 - starttime1
    #print('time1:',time1)
    
    
    
    
    
    