#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plotData(data):
    fig = plt.figure()
    plt.scatter(data[:,0],data[:,1],s=4,label='point')
    plt.xlabel = 'X'
    plt.ylabel = 'Y'
    plt.title = 'DL'
    plt.legend()
    fig.savefig('data.jpg')
    plt.show()
    plt.close(fig)

def normal(X):
    Xmean = X.mean(axis = 0)
    Xmin = X.min(axis = 0)
    Xmax = X.max(axis = 0)
    Xu = (X - Xmean)/(Xmax - Xmin)
    return Xu

def plotLine(data,W):
    fig = plt.figure()
    Dmin = data.min(axis = 0)
    Dmax = data.max(axis = 0)
    x = np.linspace(Dmin[0],Dmax[0],10)
    y = (-W[0]*x - W[2])/W[1]
    plt.plot(x,y,'r-')
    plt.scatter(data[:,0],data[:,1],s=4,label='point')
    fig.xlabel = 'X'
    fig.ylabel = 'Y'
    fig.title = 'DL'
    plt.legend()
    fig.savefig('line.jpg')
    plt.show()
    plt.close(fig)


def loss(data,W):
    AB = np.sqrt(W[0]**2 + W[1]**2)
    L = np.abs(np.dot(data,W))/AB
    l = L.mean()
    return l


def getData(weight=1,bais=0,mu=1):
    x = np.linspace(1,20,100)
    y = np.random.normal(0,mu,100) + x * weight + bais
    data = np.vstack([x,y])
    data = data.T
    return data


def DL(data,num,lr):
    W = np.random.rand(3,1)
    for i in range(num):
        l = loss(data,W)
        #点到直线的距离  L = |Ax + By + C|/sqrt( A^2 + B^2)
        AB = np.sqrt(W[0]**2 + W[1]**2)

        #由于点到直线的距离带有绝对值符号  所以在此求出 直线上方与下方的点  并在求导时使用
        flag = np.dot(data,W)/AB
        ind1 = np.where(flag > 0)
        ind2 = np.where(flag < 0)
        flag[ind1] = 1
        flag[ind2] = -1

        L = np.dot(data,W)/AB
        #求偏导
        dw = ((data*AB - L * W.T)*flag)/AB**2
        dw = dw.mean(axis = 0)
        dc = L.mean()
        dw[2] = dc
        dw = dw.reshape((3,-1))

        W = W - dw*lr

    print("loss:%f"%l)
    plotLine(data,W)
    return float(W[0]/(-W[1])),float(W[2]/(-W[1]))

if __name__ == '__main__':
    lr = 0.1
    iter_num = 10000
    #w,b = map(int,input('W,B:').split(' '))
    w,b = 1.5,1
    data = getData(w,b)
    n = data.shape[0]
    data = np.hstack([data,np.ones((n,1))])
    plotData(data)
    w = DL(data,iter_num,lr)
    print(w)


