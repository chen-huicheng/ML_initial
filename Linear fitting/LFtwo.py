#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# In[21]:


def plotData(x,y):
    fig = plt.figure()
    plt.scatter(x,y,s=4,label='point')
    plt.xlabel = 'X'
    plt.ylabel = 'Y'
    plt.title = 'dataSet'
    plt.legend()
    plt.show()
    plt.close(fig)


# In[22]:


def normal(X):
    Xmean = X.mean(axis = 0)
    Xmin = X.min(axis = 0)
    Xmax = X.max(axis = 0)
    Xu = (X - Xmean)/(Xmax - Xmin)
    return Xu


# In[23]:


def plotLine(x,y,w,b):
    fig = plt.figure()
    Dmin = x.min()
    Dmax = x.max()
    lx = np.linspace(Dmin,Dmax,10)
    ly = w * lx + b
    plt.plot(lx,ly,'r-',label='line')
    plt.scatter(x,y,s=4,label='data point')
    plt.xlabel = 'X'
    plt.ylabel = 'Y'
    plt.title = 'Linear fitting'
    plt.legend()
    plt.show()
    plt.close(fig)


# In[24]:


def loss(x,y,w,b):
    y_ = x * w + b
    l = (y - y_)**2
    return l.sum()


# In[25]:


def getData(weight=1,bais=0,mu=1):
    x = np.linspace(1,10,100)
    y = np.random.normal(0,mu,100) + x * weight + bais
    data = np.vstack([x,y])
    return data


# In[26]:


def DL(x,y,num,lr):
    W,b = 1,0
    for i in range(num):
        y_ = W * x + b
        dw = 2*(y_ - y) * x
        db = 2*(y_ - y)
        dw = dw.mean()
        db = db.mean()
        W = W - dw * lr
        b = b - db * lr
        l = loss(x,y,W,b)
    print("loss:%f"%l)
    plotLine(x,y,W,b)
    return W,b


# In[27]:


lr = 0.01
iter_num = 10000
w,b = 1.5,0.5
data = getData(w,b)
x = data[0]
y = data[1]
plotData(x,y)


# In[28]:


w,b = DL(x,y,iter_num,lr)
print(w,b)


# In[ ]:




