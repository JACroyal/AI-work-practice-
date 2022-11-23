# -*- coding: utf-8 -*-

import pandas as pd
import talib 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

stock = pd.read_csv('./csi300.csv')

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




