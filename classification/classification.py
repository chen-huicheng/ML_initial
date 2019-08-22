# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:32:56 2019

@author: Icheng
"""
import numpy as np
from openpyxl import load_workbook
import matplotlib.pyplot as plt

filename = 'data.xlsx'
lr = 0.05
iter_num=200

def loadData(filename):
    wb = load_workbook(filename)
    sheet = wb['Sheet1']
    data = []
    for row in sheet.rows:
        temp = []
        for i in row:
            temp.append(i.value)
        data.append(temp)
    return np.array(data)

def sigmoid(x):
    return (1)/(1+np.exp(-x))

def normalization(X):
    Xmean = X.mean(axis = 0)
    Xmin = X.min(axis = 0)
    Xmax = X.max(axis = 0)
    Xnorm = (X - Xmean)/(Xmax - Xmin)
    return Xnorm

def plotData(X,y,filename='temp.jpg'):
    plt_data = plt.figure(1)
    ind1 = np.where(y==1)
    ind2 = np.where(y==0)
    p1 = plt.scatter(X[ind1,0],X[ind1,1],marker='s',color='red')
    p2 = plt.scatter(X[ind2,0],X[ind2,1],marker='o',color='blue')
    plt.legend((p1,p2),('Like',"Don't like"),loc = 'upper right')
    plt_data.savefig(filename)
    plt.show(plt_data)
    plt.close(plt_data)

def show(X,y,W,b,filename='temp.jpg'):
    plt_data = plt.figure(1)
    ind1 = np.where(y==1)
    ind2 = np.where(y==0)
    p1 = plt.scatter(X[ind1,0],X[ind1,1],marker='s',color='red')
    p2 = plt.scatter(X[ind2,0],X[ind2,1],marker='o',color='blue')
    plt.legend((p1,p2),('Like',"Don't like"),loc = 'upper right')
    plt.xlabel='X'
    plt.ylabel='Y'
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    x_ = np.arange(Xmin[0],Xmax[0],0.01)
    y_ = (-b-W[0]*x_)/W[1]
    plt.plot(x_,y_)
    plt_data.savefig(filename)
    plt.show(plt_data)
    plt.close(plt_data)

def loss(y,y_):
    l = -(y*np.log(y_)+(1-y)*np.log(1-y_))
    return l.mean()

def BGD(X, y, iter_num, alpha):
    n,m = X.shape
    y = y.reshape((-1,1))
    W = np.ones((2,1))
    b = np.ones((1*1))
    for i in range(iter_num):
        out = np.dot(X,W) + b
        y_ = sigmoid(out)
        dout = y_ - y
        dw = (1.0/n)*alpha*np.dot(X.T,dout)
        W = W -dw
        db = dout.mean()
        b = b - db*alpha
    cost = loss(y,y_)
    print('cost:%f'%cost)
    show(X,y,W,b)
#    print(W,b)
    return W,b

def predict(X,y,W,b):
    out = np.dot(X,W) + b
    y_ = sigmoid(out)
    ind1 = np.where(y_ >= 0.5)
    ind2 = np.where(y_ < 0.5)
    y_[ind1] = 1
    y_[ind2] = 0
    y_ = y_.reshape(1,-1)
    result = y==y_
    return result.mean()
    
if __name__ == '__main__':
    data = loadData(filename)
    X = data[:,:2]
    y = data[:,2]
#    plotData(X,y)
    X = normalization(X)
#    plotData(X,y)
    W,b = BGD(X,y,iter_num,lr)
    pred = predict(X,y,W,b)
    print("pred:%f"%pred)