# -*- coding: utf-8 -*-
"""
姓名：陈嘉祺
学号：202003140441
"""



import pandas as pd
import talib 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score




#导入数据
stock = pd.read_csv('./csi300.csv')

#第一大题，计算收盘价的涨跌，对数据进行分类
i = 0
y_label = []
while i <= 1974:
    y = 0
    n = 1
    while n <= 5:
        if stock.iloc[i+n,4]>stock.iloc[i+n-1,4]:
            y = y + 1
        else:
            y = y + 0
        n = n + 1
    y_label.append(y)
    i = i + 1
    
#因为最后五天不具有未来五天的数据，且样本数量足够多，因此为保证数据准确，直接将其舍弃掉
stock = stock.drop(stock.index[[1975,1976,1977,1978,1979]])
print(y_label)
stock['ylabel'] = y_label
print(stock.head())
       

#第二大题，运用talib模块计算股票各项指标并导入到dataframe中
SMA = talib.SMA(stock['close'],5)
MA= talib.MA(stock['close'], timeperiod = 30, matype=0)
EMA = talib.EMA(np.array(stock['close']), timeperiod=6)
AD = talib.AD(stock['high'], stock['low'], stock['close'],stock['volume'])
OBV = talib.OBV(stock['close'], stock['volume'])
ADX = talib.ADX(stock['high'].values, stock['low'].values, stock['close'], 
timeperiod=14)
CCI = talib.CCI(stock['high'].values, stock['low'].values,stock['close'], 
timeperiod=14)
MFI = talib.MFI(stock['high'].values, stock['low'].values, stock['close'], 
stock['volume'], timeperiod=14)
RSI = talib.RSI(stock['close'], timeperiod=14)

type_all = ['SMA','MA','EMA','AD','OBV','ADX','CCI','MFI','RSI']

for i in range(len(type_all)):
    stock[type_all[i]] = locals()[type_all[i]]
print(stock.head())
#第三大题，选取适当的分类模型完成分类指标
'''
常见的模型分类器有SVM、朴素贝叶斯分类器、决策树、随机森林等，对于本次作业我选择的是随机森林模型，接下来我将说明选择随机森林的原因以及
它与其他模型之间的优劣对比（针对本题）
1.决策树和随机森林
首先介绍决策树（这里主要指CART，即分类与回归树），决策树是一个分层结构，它有多个节点，从根开始不断分叉延伸成树状，故名决策树，它可以
为每个节点都赋予一个层次数。它用一组嵌套的规则进行预测，在树的每个决策节点处，根据判断结果进入一个分叉，反复执行这种操作直到到达叶子节点，
得到预测结果。从这点上看，决策树天然是一种分类模型，尤其是天然支持多分类问题的特性，对于解决本题有很大优势。
然而决策树也有其弊端，就是容易过拟合，为了规避其容易过拟合的缺点同时利用其非常适用于多分类问题的特性，于是引入随机森林，随机森林是一种集成
方法，通过集成多个评估器形成累积的效果，形象的说，就是让多个评估器对都数据进行评估，然后进行投票，票数最高的结果会被采纳，因为决策树容易产
生过拟合的情况，那么随机森林的集成评估算法正好可以将其求均值从而减小过拟合带来的影响，得到更好的分类结果
2.支持向量机SVM
SVM也是一种优秀的分类模型，但区别是它是一种典型的线性分类器，天然适合二分类问题，而本题是多分类问题，如果想用SVM模型方法上也能说得通，可以
用一对剩余方法或者一对一方法，且不说其步骤本身是随机森林的好几倍，在处理数据时出错的几率也会增大本质上还是将多分类转化成二分类来求解，因此，SVM虽
然可以用来处理本题，但显然，它并不是最优解
3.朴素贝叶斯分类器
朴素贝叶斯分类器是一种较为简单的分类算法模型，它的基本原理就是贝叶斯公式，核心思想很简单直接，简单地说，就是利用先验概率和条件概率计算后验概率，
后验概率最大的就是结果类。朴素贝叶斯分类器原理比较简单，需要的参数也很少，很适合用来处理简单的分类问题，但是它对变量有个前提假设，那就是特征向量
的各分量之间相互独立，而本题要求的变量的来源都是同一支股票，变量之间肯定是有关联性的，所以虽然朴素贝叶斯是一种简便好用的分类模型，但它对于本题并不适用
'''

#第四大题
    #第1小题，获取训练集、测试机的 X，y 的数据

#数据处理，将含有not a number数值的行给删除，原因是含nan数据在整体数据中占比很小，为保证后续分类的准确性，将其删去
dataset = stock.dropna(axis=0)
print(dataset.head())

#提取需要的特征集，并转化成ndarray以便后续处理
dataset_use = dataset.drop(labels=['Date','open','high','low','close','volume','ylabel'],axis=1)
print(dataset_use.head())
X = np.array(dataset_use)
print(X)

#提取标签，并转化为ndarray
y = np.array(dataset.loc[:,'ylabel'])
print(y)

#利用model_selection.train_test_split()函数8:2划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
print(X_train,X_test,y_train,y_test)
data_dict = {'特征训练集':X_train,'特征测试集':X_test,
              '标签训练集':y_train,'标签测试集':y_test}
for i in range(0,4):
    print('{}为：'.format(list(data_dict.keys())[i]))
    print(list(data_dict.values())[i])


  #第2小题，利用sklearn包编写预测函数，默认参数设置
  
#利用导入的随机森林模型类，建立模型实例
model = RandomForestClassifier()

#进行五折交叉验证，并求出验证平均值
score = cross_val_score(model,X_train,y_train,cv=5)
print(score.mean())

#拟合数据集，并对测试集进行预测，计算模型准确率
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
predict = accuracy_score(y_test, y_pred)
print(predict)

#定义可以得到模型准确率的详细函数
def predict_function(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('预测类别：')
    print(y_pred)
    print('模型准确率为：')
    print(accuracy_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)
predict_function(model, X_train, y_train, X_test, y_test)

#定义简单函数，仅用来返回模型准确率
def pred_easy(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test,y_pred)




   #第3小题，选取 1-2 个此模型的重要参数。以 Score 函数为最终评估指标，分析选取的
   #参数在一定的取值范围内的变动对模型样本内和样本外预测结果的影响，并画图

    #(1)选取n_estimators为自变量，预测准确率为因变量，并对样本内和样本外数据分别绘制图像
estim_list_in = []
for i in range(1,201):
    forest = RandomForestClassifier(n_estimators=i,n_jobs=-1) 
    score = pred_easy(forest, X_train, y_train, X_train, y_train)  
    estim_list_in.append(score)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=[12,6])
plt.ylim([0,1])
plt.title('样本内树的数量与模型准确率关系曲线')
plt.plot(range(1,201),estim_list_in)
plt.savefig('./estim_in.png')
plt.show()
    
estim_list_out = []
for i in range(1,201):
    forest = RandomForestClassifier(n_estimators=i,n_jobs=-1) 
    score = pred_easy(forest, X_train, y_train, X_test, y_test)  
    estim_list_out.append(score)
#print(max(estim_list),estim_list.index(max(estim_list)))

plt.figure(figsize=[12,6])
plt.ylim([0.450,0.556])
plt.title('样本外树的数量与模型准确率关系曲线')
plt.plot(range(1,201),estim_list_out)
plt.savefig('./estim_out.png')
plt.show()

    #(2)选取min_samples_leaf为自变量，预测准确率为因变量
leaf_list_in = []
for i in range(1,101):
    forest = RandomForestClassifier(min_samples_leaf=i,n_jobs=-1) 
    score = pred_easy(forest, X_train, y_train, X_train, y_train)  
    leaf_list_in.append(score)

plt.figure(figsize=[12,6])
plt.xlim([10,101])
plt.ylim([0,1])
plt.title('样本内叶子节点上应有的最小样例数与模型准确率关系曲线')
plt.plot(range(1,101),leaf_list_in)
plt.savefig('./leaf_in.png')
plt.show()

leaf_list_out = []
for i in range(1,101):
    forest = RandomForestClassifier(min_samples_leaf=i,n_jobs=-1) 
    score = pred_easy(forest, X_train, y_train, X_test, y_test)  
    leaf_list_out.append(score)

plt.figure(figsize=[12,6])
plt.xlim([1,101])
plt.ylim([0.301,0.600])
plt.title('样本外叶子节点上应有的最小样例数与模型准确率关系曲线')
plt.plot(range(1,101),leaf_list_out)
plt.savefig('./leaf_out.png')
plt.show()




