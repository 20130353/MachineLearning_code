# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-14
# file: SVM_class.py
# description:

# code reference link: https://www.youtube.com/watch?v=bP0AhC3B6fY
# code reference link: https://blog.csdn.net/sinat_33829806/article/details/78388025

import random
import math
import copy
import numpy as np

class SVM_class:
      def __init__(self, data, label, kernel, max_iter, C, epsilon):

            self.N, self.dims = np.shape(data)
            self.C = C  # pennalty
            self.kernel = kernel
            self.maxIter = max_iter
            self.epsilon=epsilon #e_pu_si_long
            self.a = np.zeros((1,np.shape(data)[0])) # alpha init is [0](1*N)
            self.w = np.zeros((1,np.shape(data)[1])) # w init is [0](1*dim)
            self.eCache = [[0,0] for i in range(len(data))] # cache [[0,]](N*(1*dim))
            self.b = 0 # b init is 0
            self.xL= data # x data
            self.yL= label # y label

      def train(self):
            #support_Vector=self.__SMO()
            self.__SMO()
            self.__update()

      def __kernel(self,A,B):
            res=0
            if self.kernel=='Line':
                  res=self.__Tdot(A,B)
            elif self.kernel[0]=='Gauss':
                  K=0
                  for m in range(len(A)):
                       K+=(A[m]-B[m])**2
                  res=math.exp(-0.5*K/(self.kernel[1]**2))
            return res


      def __Tdot(self,A,B):
            res = 0
            for k in range(len(A)):
                  res += A[k]*B[k]
            return res


      def __SMO(self):
            support_Vector=[]
            self.a = [0 for i in range(self.N)]
            pre_a=copy.deepcopy(self.a)
            # interation times
            for it in range(self.maxIter):
                  flag=1
                  # interation alpha
                  for i in range(len(self.xL)):

                        #------------------- Step1. solve alpha 2    -------------------------
                        diff=0

                        # ---- Step1.1 solve w and b  ----
                        self.__update()

                        # ---- Step1.2 calculate Error of yhat1 and y1  ----
                        Ei=self.__calE(self.xL[i],self.yL[i])


                        # ---- Step1.3 calculate Error of yhat1 and y1 and j ----
                        j,Ej=self.__chooseJ(i,Ei)

                        # ---- Step1.4 solve bottom and top limitions (L/H) ----
                        (L,H)=self.__calLH(pre_a,j,i)

                        # ---- Step1.5 solve alpha2 ----
                        kij=self.__kernel(self.xL[i],self.xL[i])+self.__kernel(self.xL[j],\
                                        self.xL[j])-2*self.__kernel(self.xL[i],self.xL[j])

                        if(kij==0):
                              continue
                        self.a[j] = pre_a[j] + float(1.0*self.yL[j]*(Ei-Ej))/kij

                        # ---- Step1.6 check alpha2 in [L,H]  ----
                        self.a[j] = min(self.a[j], H)
                        self.a[j] = max(self.a[j], L)

                        self.eCache[j]=[1,self.__calE(self.xL[j],self.yL[j])]

                        # ------------------- Step2. solve alpha 1    -------------------------
                        self.a[i] = pre_a[i]+self.yL[i]*self.yL[j]*(pre_a[j]-self.a[j])
                        self.eCache[i]=[1,self.__calE(self.xL[i],self.yL[i])]
                        diff=sum([abs(pre_a[m]-self.a[m]) for m in range(len(self.a))])
                        if diff < self.epsilon:
                              flag=0
                        pre_a=copy.deepcopy(self.a)
                  if flag==0:
                        print(it,"break")
                        break

            #return support_Vector

      def __chooseJ(self,i,Ei):

            self.eCache[i]=[1,Ei]
            chooseList=[]

            # get j when E[j] is the largest Error
            for p in range(len(self.eCache)):
                  if self.eCache[p][0]!=0 and p!=i:
                        chooseList.append(p)
            if len(chooseList)>1:
                  delta_E=0
                  maxE=0
                  j=0
                  Ej=0
                  for k in chooseList:
                        Ek=self.__calE(self.xL[k],self.yL[k])
                        delta_E=abs(Ek-Ei)
                        if delta_E>maxE:
                              maxE=delta_E
                              j=k
                              Ej=Ek
                  return j,Ej
            else:
                  # init state
                  j=self.__randJ(i)
                  Ej=self.__calE(self.xL[j],self.yL[j])
                  return j,Ej

      def __randJ(self,i):
            # random select j
            j=i
            while(j==i):
                  j=random.randint(0,len(self.xL)-1)
            return j

      def __calLH(self,pre_a,j,i):
            if(self.yL[j]!= self.yL[i]):
                  return (max(0,pre_a[j]-pre_a[i]),min(self.C,self.C-pre_a[i]+pre_a[j]))
            else:
                  return (max(0,-self.C+pre_a[i]+pre_a[j]),min(self.C,pre_a[i]+pre_a[j]))

      def __calE(self,x,y):

            # predict the yhat
            y_,q=self.predict(x)
            # return yhat - true y
            return y_-y

      def __calW(self):
            # w = alpha_i * x_i * y_i (i is dimension)
            self.w = [0 for i in range(self.dims)]
            for i in range(self.N):
                  for j in range(len(self.w)):
                        self.w[j]+=self.a[i]*self.yL[i]*self.xL[i][j]

      def __update(self):

            # step1. get w: w = alpha * x * y
            self.__calW()

            # step2. get b
            maxf1=-99999
            min1=99999
            for k in range(self.N):
                  # summation of wi*xi, i is the dimensions
                  # xL is the length of x data
                  y_v=self.__Tdot(self.w,self.xL[k])

                  if self.yL[k]==-1:
                        if y_v>maxf1:
                              maxf1=y_v
                  else:
                        if y_v<min1:
                              min1=y_v

            # get b through w
            self.b=-0.5*(maxf1+min1)

      def predict(self,testData):
            pre_value=0
            for i in range(self.N):
                  pre_value += self.a[i]*self.yL[i]*self.__kernel(self.xL[i],testData)
            pre_value += self.b

            print(pre_value,"pre_value")

            if pre_value<0:
                  y=-1
            else:
                  y=1
            return y,abs(pre_value-0)


if __name__ == '__main__':
    # because the y in [-1,+1], the test file is not suitable for it
    x = [[1, 1], [2, 1], [1, 0], [3, 7], [4, 8], [4, 10]]
    y = [1, 1, 1, -1, -1, -1]

    svm = SVM_class(data=x, label=y, kernel='Line', max_iter=1000, C=0.02, epsilon=0.001)
    svm.train()
    print(svm.predict([4, 0]))

    print('alpha is\n', svm.a)
    print('w is\n', svm.w)
    print('b is \n', svm.b)







